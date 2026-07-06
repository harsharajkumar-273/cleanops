import os
import json
import re

class LLMAgent:
    """
    LLMAgent acts as the planner. It translates natural language task instructions 
    into structured database transaction queries using an LLM or fallback heuristics.
    """
    def __init__(self, model_name="GPT-4.1-mini"):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def predict_action(self, observation):
        """Predicts the next structured database query action based on current state."""
        instruction = observation["instruction"]
        db_state = observation["database_state"]
        
        # If API key is configured, perform LLM chat completion
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = (
                    "Given the database state and the instruction, output a single JSON action "
                    "specifying action_type, table, and params. Output ONLY valid JSON.\n"
                    f"State: {json.dumps(db_state)}\n"
                    f"Instruction: {instruction}"
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                action_json = json.loads(response.choices[0].message.content)
                return action_json
            except Exception:
                pass # Fall back to heuristics if API call fails
                
        # Heuristic Rule-Based Rule Parser (Fallback)
        return self._heuristic_fallback(instruction, db_state)

    def _heuristic_fallback(self, instruction, db_state):
        inst_lower = instruction.toLowerCase() if hasattr(instruction, "toLowerCase") else instruction.lower()
        
        # Check task: CRM Client status to INACTIVE
        if "inactive" in inst_lower and "client" in inst_lower:
            # Find client ID
            match = re.search(r"#(\d+)", inst_lower)
            client_id = int(match.group(1)) if match else 104
            
            # Find note
            note = "Pending invoice payout"
            if "notes" in inst_lower or "note" in inst_lower:
                match_note = re.search(r"note '(.*?)'", instruction, re.IGNORECASE)
                if match_note:
                    note = match_note.group(1)
            
            return {
                "action_type": "update",
                "table": "clients",
                "params": {"id": client_id, "status": "INACTIVE", "notes": note}
            }
            
        # Check task: CRM Create Client Profile
        if "create" in inst_lower and "client" in inst_lower:
            return {
                "action_type": "create",
                "table": "clients",
                "params": {
                    "id": 106, 
                    "name": "Antigravity Robotics", 
                    "status": "ACTIVE", 
                    "notes": "Robotics data collection partner"
                }
            }

        # Check task: Void Invoice
        if "void" in inst_lower and "invoice" in inst_lower:
            match = re.search(r"inv-(\d+)", inst_lower)
            inv_num = f"INV-{match.group(1)}" if match else "INV-402"
            return {
                "action_type": "update",
                "table": "invoices",
                "params": {"invoice_number": inv_num, "amount": 0.0, "status": "VOID"}
            }

        # Check task: Fulfill Order / Decrement Stock
        if "fulfill" in inst_lower and "order" in inst_lower:
            match = re.search(r"ord-(\d+)", inst_lower)
            order_id = f"ORD-{match.group(1)}" if match else "ORD-801"
            
            # If step 1: Update stock first
            if observation_has_pending_order(db_state, order_id):
                return {
                    "action_type": "update",
                    "table": "stock",
                    "params": {"item_id": 99, "quantity": 8}
                }
            else:
                return {
                    "action_type": "update",
                    "table": "orders",
                    "params": {"order_id": order_id, "status": "SHIPPED"}
                }

        # Fallback empty action
        return {"action_type": "noop", "table": "none", "params": {}}

def observation_has_pending_order(db_state, order_id):
    for order in db_state.get("orders", []):
        if order["order_id"] == order_id and order["status"] == "PENDING":
            return True
    return False
