import copy

class OpenEnv:
    """
    OpenEnv is a sandboxed evaluation environment representing business databases.
    It holds current table states and evaluates agent actions against target oracle states.
    """
    def __init__(self):
        self.current_state = {}
        self.target_state = {}
        self.instruction = ""
        self.task_id = ""
        self.step_count = 0
        self.max_steps = 5

    def reset(self, task_dict):
        """Resets environment state to the start configuration of a task."""
        self.task_id = task_dict.get("task_id", "")
        self.instruction = task_dict.get("instruction", "")
        self.current_state = copy.deepcopy(task_dict.get("initial_state", {}))
        self.target_state = copy.deepcopy(task_dict.get("target_state", {}))
        self.step_count = 0
        return self.get_observation()

    def get_observation(self):
        """Returns the current state representation visible to the agent."""
        return {
            "instruction": self.instruction,
            "database_state": self.current_state,
            "step_count": self.step_count
        }

    def step(self, action):
        """
        Executes a database transaction action.
        
        Action format example:
        {
            "action_type": "update" | "create" | "delete",
            "table": "clients" | "invoices" | "orders" | "stock",
            "params": { ... }
        }
        """
        self.step_count += 1
        info = {"status": "success", "error": None}
        
        action_type = action.get("action_type")
        table = action.get("table")
        params = action.get("params", {})

        if not action_type or not table:
            info["status"] = "failed"
            info["error"] = "Invalid action: action_type and table are required."
            return self.get_observation(), 0.0, False, info

        # Execute Transaction
        try:
            if table not in self.current_state and action_type != "create":
                # Create table dynamic fallback
                self.current_state[table] = []

            # Handle CRM clients table
            if table == "clients":
                if action_type == "update":
                    client_id = params.get("id")
                    updated = False
                    for client in self.current_state["clients"]:
                        if client["id"] == client_id:
                            for k, v in params.items():
                                if k != "id":
                                    client[k] = v
                            updated = True
                    if not updated:
                        info["status"] = "failed"
                        info["error"] = f"Client ID {client_id} not found."
                elif action_type == "create":
                    # Generate ID if missing
                    if "id" not in params:
                        max_id = max([c["id"] for c in self.current_state.get("clients", [{"id": 100}])])
                        params["id"] = max_id + 1
                    self.current_state.setdefault("clients", []).append(params)
            
            # Handle Invoices table
            elif table == "invoices":
                if action_type == "update":
                    inv_num = params.get("invoice_number")
                    updated = False
                    for invoice in self.current_state["invoices"]:
                        if invoice["invoice_number"] == inv_num:
                            for k, v in params.items():
                                if k != "invoice_number":
                                    invoice[k] = v
                            updated = True
                    if not updated:
                        info["status"] = "failed"
                        info["error"] = f"Invoice {inv_num} not found."

            # Handle Stock / Orders table
            elif table == "stock":
                if action_type == "update":
                    item_id = params.get("item_id")
                    if isinstance(self.current_state.get("stock"), dict) and self.current_state["stock"].get("item_id") == item_id:
                        for k, v in params.items():
                            if k != "item_id":
                                self.current_state["stock"][k] = v
                    elif isinstance(self.current_state.get("stock"), list):
                        for item in self.current_state["stock"]:
                            if item["item_id"] == item_id:
                                for k, v in params.items():
                                    if k != "item_id":
                                        item[k] = v

            elif table == "orders":
                if action_type == "update":
                    order_id = params.get("order_id")
                    for order in self.current_state["orders"]:
                        if order["order_id"] == order_id:
                            for k, v in params.items():
                                if k != "order_id":
                                    order[k] = v

            else:
                info["status"] = "failed"
                info["error"] = f"Unsupported table schema: {table}"

        except Exception as e:
            info["status"] = "failed"
            info["error"] = f"Transaction runtime exception: {str(e)}"

        # Compute Done
        done = (self.current_state == self.target_state) or (self.step_count >= self.max_steps)
        
        return self.get_observation(), done, info
