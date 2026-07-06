def calculate_reward(current_state, target_state, action, info):
    """
    Computes a dense reward signal providing intermediate partial credits 
    based on syntactic correctness, schema targeting, and key alignment.
    """
    # 0. If execution encountered a transaction error, reward is 0
    if info.get("status") == "failed":
        return 0.0

    # 1. Start with baseline reward for error-free execution
    reward = 0.2
    
    action_type = action.get("action_type")
    table = action.get("table")
    params = action.get("params", {})

    # 2. Check if the action targeted a table that exists in target_state
    if table in target_state:
        reward += 0.25  # Reaches 0.45
        
        # 3. Check key-level alignment
        key_matched = False
        if table == "clients":
            client_id = params.get("id")
            target_ids = [c["id"] for c in target_state.get("clients", [])]
            if client_id in target_ids:
                key_matched = True
        elif table == "invoices":
            inv_num = params.get("invoice_number")
            target_invs = [i["invoice_number"] for i in target_state.get("invoices", [])]
            if inv_num in target_invs:
                key_matched = True
        elif table == "stock":
            item_id = params.get("item_id")
            if isinstance(target_state.get("stock"), dict) and target_state["stock"].get("item_id") == item_id:
                key_matched = True
        elif table == "orders":
            order_id = params.get("order_id")
            target_orders = [o["order_id"] for o in target_state.get("orders", [])]
            if order_id in target_orders:
                key_matched = True

        if key_matched:
            reward += 0.2  # Reaches 0.65

    # 4. Check if final state matches the target state exactly (Full credit check)
    if current_state == target_state:
        return 1.0

    return min(reward, 0.95)
