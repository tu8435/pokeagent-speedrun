import random
import logging
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
from agent.system_prompt import system_prompt

# Set up module logging
logger = logging.getLogger(__name__)

def action_step(memory_context, current_plan, latest_observation, frame, state_data, recent_actions, vlm):
    """
    Decide and perform the next action button(s) based on memory, plan, observation, and comprehensive state.
    Returns a list of action buttons as strings.
    """
    # Get formatted state context and useful summaries
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    movement_options = get_movement_options(state_data)
    party_health = get_party_health_summary(state_data)
    
    logger.info("[ACTION] Starting action decision")
    logger.info(f"[ACTION] State: {state_summary}")
    logger.info(f"[ACTION] Party health: {party_health['healthy_count']}/{party_health['total_count']} healthy")
    if movement_options:
        logger.info(f"[ACTION] Movement options: {movement_options}")
    
    # Build enhanced action context
    action_context = []
    
    # Extract key info for context
    game_data = state_data.get('game', {})
    
    # Battle vs Overworld context
    if game_data.get('in_battle', False):
        action_context.append("=== BATTLE MODE ===")
        battle_info = game_data.get('battle_info', {})
        if battle_info:
            if 'player_pokemon' in battle_info:
                player_pkmn = battle_info['player_pokemon']
                action_context.append(f"Your Pokemon: {player_pkmn.get('species_name', player_pkmn.get('species', 'Unknown'))} (Lv.{player_pkmn.get('level', '?')}) HP: {player_pkmn.get('current_hp', '?')}/{player_pkmn.get('max_hp', '?')}")
            if 'opponent_pokemon' in battle_info:
                opp_pkmn = battle_info['opponent_pokemon']
                action_context.append(f"Opponent: {opp_pkmn.get('species_name', opp_pkmn.get('species', 'Unknown'))} (Lv.{opp_pkmn.get('level', '?')}) HP: {opp_pkmn.get('current_hp', '?')}/{opp_pkmn.get('max_hp', '?')}")
    else:
        action_context.append("=== OVERWORLD MODE ===")
        
        # Movement options from utility
        if movement_options:
            action_context.append("Movement Options:")
            for direction, description in movement_options.items():
                action_context.append(f"  {direction}: {description}")
    
    # Party health summary
    if party_health['total_count'] > 0:
        action_context.append("=== PARTY STATUS ===")
        action_context.append(f"Healthy Pokemon: {party_health['healthy_count']}/{party_health['total_count']}")
        if party_health['critical_pokemon']:
            action_context.append("Critical Pokemon:")
            for critical in party_health['critical_pokemon']:
                action_context.append(f"  {critical}")
    
    # Recent actions context
    if recent_actions:
        action_context.append(f"Recent Actions: {', '.join(list(recent_actions)[-5:])}")
    
    context_str = "\n".join(action_context)
    
    action_prompt = f"""
    ★★★ COMPREHENSIVE GAME STATE DATA ★★★
    
    {state_context}
    
    ★★★ ENHANCED ACTION CONTEXT ★★★
    
    {context_str}
    
    ★★★ ACTION DECISION TASK ★★★
    
    You are the agent playing Pokemon Emerald with a speedrunning mindset. Make quick, efficient decisions.
    
    Memory Context: {memory_context}
    Current Plan: {current_plan if current_plan else 'No plan yet'}
    Latest Observation: {latest_observation}
    
    Based on the comprehensive state information above, decide your next action(s):
    
    BATTLE STRATEGY:
    - If in battle: Choose moves strategically based on type effectiveness and damage
    - Consider switching pokemon if current one is weak/low HP
    - Use items if pokemon is in critical condition
    
    NAVIGATION STRATEGY:
    - Use movement options analysis above for efficient navigation
    - Avoid blocked tiles (marked as BLOCKED)
    - Consider tall grass: avoid if party is weak, seek if need to train/catch
    - Navigate around water unless you have Surf
    - Use coordinates to track progress toward objectives
    
    MENU/DIALOGUE STRATEGY:
    - If in dialogue: A, A, A to advance quickly
    - If in menu: Navigate to desired option efficiently
    - If stuck in menu: B to cancel/back out
    
    HEALTH MANAGEMENT:
    - If pokemon are low HP/fainted, head to Pokemon Center
    - If no healthy pokemon, prioritize healing immediately
    - Consider terrain: avoid wild encounters if party is weak
    
    EFFICIENCY RULES:
    1. Output sequences of actions when you know what's coming (e.g., "RIGHT, RIGHT, RIGHT, A" to enter a door)
    2. For dialogue: "A, A, A, A, A" to mash through
    3. For movement: repeat directions based on movement options (e.g., "UP, UP, UP, UP" if UP shows "Normal path")
    4. If uncertain, output single action and reassess
    5. Use traversability data: move toward open paths, avoid obstacles
    
    Valid buttons: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, L, R
    - A: Interact, confirm, attack
    - B: Cancel, back, run (with running shoes)
    - START: Main menu
    - SELECT: Use registered item
    - Directional: Move, navigate menus (use movement options above)
    - L/R: Cycle through options in some menus
    
    Return ONLY the button name(s) as a comma-separated list, nothing else.
    Maximum 10 actions in sequence. Avoid repeating same button more than 6 times.
    """
    
    action_response = vlm.get_text_query(system_prompt + action_prompt, "ACTION").strip().upper()
    valid_buttons = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']
    
    # Split the response by commas and clean up
    actions = [btn.strip() for btn in action_response.split(',') if btn.strip() in valid_buttons]
    
    # Limit to maximum 10 actions and prevent excessive repetition
    actions = actions[:10]
    
    # If no valid actions found, make intelligent default based on state
    if not actions:
        if game_data.get('in_battle', False):
            actions = ['A']  # Attack in battle
        elif party_health['total_count'] == 0:
            actions = ['A', 'A', 'A']  # Try to progress dialogue/menu
        else:
            actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]  # Random exploration
    
    logger.info(f"[ACTION] Actions decided: {', '.join(actions)}")
    return actions 