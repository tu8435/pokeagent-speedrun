#!/usr/bin/env python3
"""
Fixed Simple Pokemon Emerald server - handles FastAPI and pygame properly
"""

import os
import pygame
import numpy as np
import time
import base64
import io
import signal
import sys
import asyncio
import threading
import json
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import socketio     # for streamer setup

from pokemon_env.emulator import EmeraldEmulator
from pokemon_env.emerald_utils import (
    SYSTEM_FLAGS_START, FLAG_VISITED_LITTLEROOT_TOWN, FLAG_VISITED_OLDALE_TOWN,
    FLAG_VISITED_PETALBURG_CITY, FLAG_VISITED_RUSTBORO_CITY, FLAG_VISITED_DEWFORD_TOWN,
    FLAG_VISITED_SLATEPORT_CITY, FLAG_VISITED_MAUVILLE_CITY, FLAG_BADGE01_GET,
    FLAG_BADGE02_GET, FLAG_BADGE03_GET, FLAG_SYS_POKEMON_GET, FLAG_SYS_POKEDEX_GET,
    FLAG_DEFEATED_RUSTBORO_GYM, FLAG_DEFEATED_DEWFORD_GYM, FLAG_DEFEATED_MAUVILLE_GYM
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
env = None
running = True
step_count = 0
agent_step_count = 0  # Track agent steps separately from frame steps
current_obs = None
fps = 60

# Pygame display
screen_width = 480  # 240 * 2 (upscaled)
screen_height = 320  # 160 * 2 (upscaled)
screen = None
font = None
clock = None

# Threading locks for thread safety
obs_lock = threading.Lock()
step_lock = threading.Lock()
memory_lock = threading.Lock()  # New lock for memory operations to prevent race conditions

# Button mapping
button_map = {
    pygame.K_z: 'A',
    pygame.K_x: 'B', 
    pygame.K_RETURN: 'START',
    pygame.K_RSHIFT: 'SELECT',
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
}

# Milestone tracking is now handled by the emulator

# FastAPI app
app = FastAPI(
    title="Pokemon Emerald Simple Server",
    description="Simple server with pygame display and FastAPI endpoints",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class ActionRequest(BaseModel):
    buttons: list = []  # List of button names: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT

class GameStateResponse(BaseModel):
    screenshot_base64: str
    step_number: int
    resolution: list  # [width, height]
    status: str

class ComprehensiveStateResponse(BaseModel):
    visual: dict
    player: dict
    game: dict
    map: dict
    step_number: int
    status: str

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    if env:
        env.stop()
    pygame.quit()
    sys.exit(0)

def setup_environment():
    """Initialize the emulator"""
    global env, current_obs
    
    try:
        rom_path = "Emerald-GBAdvance/rom.gba"
        if not os.path.exists(rom_path):
            raise RuntimeError(f"ROM not found at {rom_path}")
        
        env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()
        
        screenshot = env.get_screenshot()
        if screenshot:
            with obs_lock:
                current_obs = np.array(screenshot)
        else:
            with obs_lock:
                current_obs = np.zeros((env.height, env.width, 3), dtype=np.uint8)

        print("Emulator initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize emulator: {e}")
        return False

def handle_input(manual_mode=False):
    """Handle keyboard input and convert to game actions"""
    global recent_button_presses
    actions_pressed = []
    
    if not manual_mode:
        # Server mode - no keyboard input
        return True, []
    
    # Manual mode - handle keyboard input
    # This handles continuous key presses
    keys = pygame.key.get_pressed()
    for key, button in button_map.items():
        if keys[key]:
            actions_pressed.append(button)

    # This handles single key events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, []
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False, []
            elif event.key == pygame.K_s:
                save_screenshot()
            elif event.key == pygame.K_r:
                reset_game()
            elif event.key == pygame.K_1:
                # Use currently loaded state file if one exists, otherwise use default
                save_file = env._current_state_file if env._current_state_file else "server/simple_test.state"
                env.save_state(save_file)
            elif event.key == pygame.K_2:
                # Use currently loaded state file if one exists, otherwise use default
                load_file = env._current_state_file if env._current_state_file else "server/simple_test.state"
                env.load_state(load_file)
    
    # Track manual button presses for the action queue display
    if actions_pressed:
        current_time = time.time()
        for button in actions_pressed:
            # Avoid duplicate consecutive buttons within 100ms
            should_add = True
            if recent_button_presses:
                last_entry = recent_button_presses[-1]
                if (last_entry["button"] == button and 
                    current_time - last_entry["timestamp"] < 0.1):  # 100ms threshold
                    should_add = False
            
            if should_add:
                recent_button_presses.append({
                    "button": button,
                    "timestamp": current_time
                })
        
        # Keep only last 50 button presses to avoid memory issues
        if len(recent_button_presses) > 50:
            recent_button_presses = recent_button_presses[-50:]
    
    return True, actions_pressed

def step_environment(actions_pressed):
    """Take a step in the environment with comprehensive locking for race condition prevention"""
    global current_obs
    
    # Use memory_lock to prevent race conditions with state reading during area transitions
    with memory_lock:
        with step_lock:
            env.run_frame_with_buttons(actions_pressed)
            
            # IMPROVED AREA TRANSITION DETECTION: Use proper _check_area_transition method
            if hasattr(env, 'memory_reader') and env.memory_reader:
                try:
                    # Use the built-in area transition detection
                    transition_detected = env.memory_reader._check_area_transition()
                    
                    if transition_detected:
                        logger.info("Area transition detected by _check_area_transition()")
                        
                        # Force complete cache invalidation
                        env.memory_reader.invalidate_map_cache()
                        
                        # Clear behavior cache specifically - this is the key fix
                        if hasattr(env.memory_reader, '_cached_behaviors'):
                            env.memory_reader._cached_behaviors = None
                        if hasattr(env.memory_reader, '_cached_behaviors_map_key'):
                            env.memory_reader._cached_behaviors_map_key = None
                        
                        # Clear memory region cache for EWRAM (where map buffer lives)
                        if hasattr(env.memory_reader, '_mem_cache'):
                            env.memory_reader._mem_cache.clear()
                        
                        # Let the system naturally refresh rather than forcing immediate re-detection
                        logger.info("Caches cleared for area transition")
                    
                    # Also do a basic location name check as backup
                    current_location = env.memory_reader.read_location()
                    if hasattr(env, '_last_location_check'):
                        if current_location != env._last_location_check:
                            logger.info(f"Location name changed: {env._last_location_check} -> {current_location}")
                            # Additional cache clearing for location name changes
                            env.memory_reader.invalidate_map_cache()
                    
                    env._last_location_check = current_location
                except Exception as e:
                    logger.debug(f"Location check failed: {e}")

            screenshot = env.get_screenshot()
            if screenshot:
                with obs_lock:
                    current_obs = np.array(screenshot)

def update_display(manual_mode=False):
    """Update the display with current game state"""
    global current_obs, screen, step_count
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is not None and screen:
        obs_surface = pygame.surfarray.make_surface(obs_copy.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(obs_surface, (screen_width, screen_height))
        screen.blit(scaled_surface, (0, 0))
    
    # if manual_mode:
    #     draw_info_overlay()
    pygame.display.flip()

def draw_info_overlay():
    """Draw information overlay on the screen"""
    global screen, step_count, font
    
    if not screen or not font:
        return
        
    overlay_height = 80
    overlay = pygame.Surface((screen_width, overlay_height))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    
    # Position overlay at the bottom of the screen
    overlay_y = screen_height - overlay_height
    screen.blit(overlay, (0, overlay_y))
    
    info_lines = [
        f"Step: {step_count}",
        f"Controls: Z=A, X=B, Enter=Start, RShift=Select, Arrows=Move",
        f"Special: S=Screenshot, R=Reset, 1=Save State, 2=Load State, Esc=Quit"
    ]
    
    y_offset = overlay_y + 8
    for line in info_lines:
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 22

def save_screenshot():
    """Save current screenshot"""
    global current_obs
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is not None:
        timestamp = int(time.time())
        filename = f"simple_test_screenshot_{timestamp}.png"
        img = Image.fromarray(obs_copy)
        img.save(filename)
        print(f"Screenshot saved: {filename}")

def reset_game():
    """Reset the game and all milestones"""
    global env, step_count
    
    print("Resetting game and milestones...")
    with step_lock:
        env.initialize()
        env.milestone_tracker.reset_all()  # Reset all milestones
        step_count = 0
    print("Game and milestone reset complete")

def game_loop(manual_mode=False):
    """Main game loop - runs in main thread"""
    global running, step_count, clock
    
    if manual_mode:
        print("Starting game loop in manual mode...")
        print("Controls: Z=A, X=B, Enter=Start, RShift=Select, Arrows=Move")
        print("Special: S=Screenshot, R=Reset, 1=Save State, 2=Load State, Esc=Quit")
    else:
        print("Starting game loop in server mode...")
    
    while running:
        # Handle input
        should_continue, actions_pressed = handle_input(manual_mode)
        if not should_continue:
            break
            
        # Step environment
        step_environment(actions_pressed)
        
        # Update milestones based on current state (for manual mode too)
        # Only update milestones occasionally to avoid performance issues
        try:
            if step_count % 20 == 0:  # Only update every 20 steps
                state = env.get_comprehensive_state()
                env.check_and_update_milestones(state)
        except Exception as e:
            logger.warning(f"Error updating milestones during manual game loop: {e}")
        
        # Update display
        update_display(manual_mode)
        
        with step_lock:
            step_count += 1
        
        # Use dynamic FPS - 2x speed during dialog
        current_fps = env.get_current_fps(fps) if env else fps
        clock.tick(current_fps)

def init_pygame():
    """Initialize pygame"""
    global screen, font, clock
    
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pokemon Emerald Simple Server")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

def run_fastapi_server(port):
    """Run FastAPI server in background thread"""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

    # Create a new Socket.IO server
    sio = socketio.AsyncServer(cors_allowed_origins='*')

    # Attach the Socket.IO server to the FastAPI app
    sio_app = socketio.ASGIApp(sio, app)

    # Define event handlers for the Socket.IO server
    @sio.event
    async def connect(sid, environ):
        print('Client connected:', sid)

    @sio.event
    async def disconnect(sid):
        print('Client disconnected:', sid)

# FastAPI endpoints
@app.get("/status")
async def get_status():
    """Get server status"""
    with step_lock:
        current_step = step_count
    
    # Get current FPS (may be 4x during dialog)
    current_fps = env.get_current_fps(fps) if env else fps
    # Use cached dialog state for consistency with FPS calculation
    is_dialog = env._cached_dialog_state if env else False
    
    return {
        "status": "running",
        "step_count": current_step,
        "base_fps": fps,
        "current_fps": current_fps,
        "is_dialog": is_dialog,
        "fps_multiplier": 2 if is_dialog else 1
    }

@app.get("/screenshot")
async def get_screenshot():
    """Get current screenshot"""
    global current_obs, step_count
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is None:
        raise HTTPException(status_code=500, detail="No screenshot available")
    
    try:
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(obs_copy)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        with step_lock:
            current_step = step_count
        
        return GameStateResponse(
            screenshot_base64=img_str,
            step_number=current_step,
            resolution=[obs_copy.shape[1], obs_copy.shape[0]],
            status="running"
        )
        
    except Exception as e:
        logger.error(f"Error getting screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/action")
async def take_action(request: ActionRequest):
    """Take an action"""
    global current_obs, step_count, recent_button_presses
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Track button presses for recent actions display
        if request.buttons:
            current_time = time.time()
            for button in request.buttons:
                # Avoid duplicate consecutive buttons within 100ms
                should_add = True
                if recent_button_presses:
                    last_entry = recent_button_presses[-1]
                    if (last_entry["button"] == button and 
                        current_time - last_entry["timestamp"] < 0.1):  # 100ms threshold
                        should_add = False
                
                if should_add:
                    recent_button_presses.append({
                        "button": button,
                        "timestamp": current_time
                    })
            
            # Keep only last 50 button presses to avoid memory issues
            if len(recent_button_presses) > 50:
                recent_button_presses = recent_button_presses[-50:]
        
        # Execute action - step_environment now handles its own memory locking
        step_environment(request.buttons)
        
        with step_lock:
            step_count += 1
            current_step = step_count
        
        # Update milestones based on current state (even without stream)
        # Only update milestones occasionally to avoid performance issues
        try:
            if step_count % 10 == 0:  # Only update every 10 steps
                state = env.get_comprehensive_state()
                env.check_and_update_milestones(state)
        except Exception as e:
            logger.warning(f"Error updating milestones during action: {e}")
        
        # Get updated screenshot
        with obs_lock:
            obs_copy = current_obs.copy() if current_obs is not None else None
        
        if obs_copy is not None:
            pil_image = Image.fromarray(obs_copy)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return GameStateResponse(
                screenshot_base64=img_str,
                step_number=current_step,
                resolution=[obs_copy.shape[1], obs_copy.shape[0]],
                status="running"
            )
        else:
            raise HTTPException(status_code=500, detail="No screenshot available")
            
    except Exception as e:
        logger.error(f"Error taking action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_comprehensive_state():
    """Get comprehensive game state including visual and memory data"""
    global step_count, milestone_tracker
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Clear the comprehensive state cache to ensure fresh data
        if hasattr(env, '_cached_state'):
            delattr(env, '_cached_state')
        if hasattr(env, '_cached_state_time'):
            delattr(env, '_cached_state_time')
        
        # Get fresh comprehensive state from emulator
        # No memory lock needed - area transitions are handled in step_environment
        state = env.get_comprehensive_state()
        
        # Ensure game state is consistent with cached dialog state
        # Use the same cached dialog state as the status endpoint
        is_dialog = env._cached_dialog_state if env else False
        if is_dialog:
            state["game"]["game_state"] = "dialog"
        else:
            # Force overworld if not in dialog (respect 5-second timeout)
            state["game"]["game_state"] = "overworld"
        
        # Update milestones based on current state
        # Only update milestones occasionally to avoid performance issues
        if step_count % 5 == 0:  # Only update every 5 steps for state endpoint
            env.check_and_update_milestones(state)
        
        # The battle information already contains all necessary data
        # No additional analysis needed - keep it clean
        
        # Convert screenshot to base64 if available
        if state["visual"]["screenshot"]:
            buffer = io.BytesIO()
            state["visual"]["screenshot"].save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            state["visual"]["screenshot_base64"] = img_str
            # Remove the PIL image object to avoid serialization issues
            del state["visual"]["screenshot"]
        
        with step_lock:
            current_step = step_count
        
        return ComprehensiveStateResponse(
            visual=state["visual"],
            player=state["player"],
            game=state["game"],
            map=state["map"],
            step_number=current_step,
            status="running"
        )
        
    except Exception as e:
        logger.error(f"Error getting comprehensive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
def format_map_for_comparison(self, tiles, title, location, position):
    """Format map tiles for comparison with ground truth format"""
    if not tiles:
        return f"=== {title} ===\nNo tiles available\n"
    
    output = []
    output.append(f"=== {title} ===")
    output.append(f"Format: (MetatileID, Behavior, X, Y)")
    output.append(f"Map dimensions: {len(tiles)}x{len(tiles[0]) if tiles else 0}")
    output.append("")
    output.append("--- TRAVERSABILITY MAP ---")
    
    # Header with column numbers
    header = "      " + "  ".join(f"{i:2}" for i in range(len(tiles[0]) if tiles else 0))
    output.append(header)
    output.append("    " + "-" * (len(header) - 4))
    
    # Map rows
    for row_idx, row in enumerate(tiles):
        traversability_row = []
        for col_idx, tile in enumerate(row):
            if len(tile) >= 4:
                tile_id, behavior, collision, elevation = tile
                behavior_val = behavior if not hasattr(behavior, 'value') else behavior.value
                
                # Convert to traversability symbol
                if behavior_val == 0:  # NORMAL
                    symbol = "." if collision == 0 else "#"
                elif behavior_val == 1:  # SECRET_BASE_WALL
                    symbol = "#"
                elif behavior_val == 51:  # IMPASSABLE_SOUTH
                    symbol = "IM"
                elif behavior_val == 96:  # NON_ANIMATED_DOOR
                    symbol = "D"
                elif behavior_val == 101:  # SOUTH_ARROW_WARP
                    symbol = "SO"
                elif behavior_val == 105:  # ANIMATED_DOOR
                    symbol = "D"
                elif behavior_val == 134:  # TELEVISION
                    symbol = "TE"
                else:
                    symbol = "."  # Default to walkable for other behaviors
                
                # Mark player position
                if position and len(position) >= 2:
                    # Calculate if this tile is player position
                    # Player is at center of 15x15 map (position 7,7)
                    if row_idx == 7 and col_idx == 7:
                        symbol = "P"
                
                traversability_row.append(symbol)
            else:
                traversability_row.append("?")
        
        # Format row with row number
        row_str = f"{row_idx:2}: " + " ".join(f"{symbol:1}" for symbol in traversability_row)
        output.append(row_str)
    
    return "\n".join(output)
    

@app.get("/debug/memory")
async def debug_memory():
    """Debug memory reading (basic version)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}
        
        # Test basic memory access
        diagnostics = env.memory_reader.test_memory_access()
        
        # Try to read some basic data
        try:
            party_size = env.memory_reader.read_party_size()
            coordinates = env.memory_reader.read_coordinates()
            money = env.memory_reader.read_money()
            
            # Add new debugging info
            is_in_battle = env.memory_reader.is_in_battle()
            game_state = env.memory_reader.get_game_state()
            player_name = env.memory_reader.read_player_name()
            
            # Add battle detection debugging
            try:
                battle_addr = env.memory_reader.IN_BATTLE_BIT_ADDR
                battle_raw_value = env.memory_reader._read_u8(battle_addr)
                battle_mask = env.memory_reader.IN_BATTLE_BITMASK
                battle_result = (battle_raw_value & battle_mask) != 0
            except Exception as e:
                battle_raw_value = None
                battle_mask = None
                battle_result = None
            
            diagnostics.update({
                'party_size': party_size,
                'coordinates': coordinates,
                'money': money,
                'is_in_battle': is_in_battle,
                'game_state': game_state,
                'player_name': player_name,
                'battle_detection': {
                    'address': f'0x{battle_addr:08x}' if 'battle_addr' in locals() else 'unknown',
                    'raw_value': f'0x{battle_raw_value:02x}' if battle_raw_value is not None else 'error',
                    'mask': f'0x{battle_mask:02x}' if battle_mask is not None else 'unknown',
                    'result': battle_result
                },
                'working_reads': True
            })
        except Exception as read_error:
            diagnostics['read_error'] = str(read_error)
            diagnostics['working_reads'] = False
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error debugging memory: {e}")
        return {"error": str(e)}

@app.get("/debug/memory/comprehensive")
async def debug_memory_comprehensive():
    """Comprehensive memory reading test with detailed diagnostics"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Use the comprehensive memory testing method
        test_results = env.test_memory_reading()
        return test_results
        
    except Exception as e:
        logger.error(f"Error running comprehensive memory test: {e}")
        return {"error": str(e)}

@app.get("/debug/memory/dump")
async def debug_memory_dump(start: int = 0x02000000, length: int = 0x1000):
    """Dump raw memory from the emulator"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}
        
        # Read raw memory bytes
        memory_bytes = env.memory_reader._read_bytes(start, length)
        
        # Convert to hex string for easy viewing
        hex_data = memory_bytes.hex()
        
        # Also try to decode as text using Pokemon Emerald character mapping
        try:
            from pokemon_env.emerald_utils import EmeraldCharmap
            charmap = EmeraldCharmap()
            decoded_text = charmap.decode(memory_bytes)
        except:
            decoded_text = "Could not decode as text"
        
        return {
            "start_address": f"0x{start:08X}",
            "length": length,
            "hex_data": hex_data,
            "decoded_text": decoded_text,
            "raw_bytes": [b for b in memory_bytes[:100]]  # First 100 bytes as numbers
        }
        
    except Exception as e:
        logger.error(f"Error dumping memory: {e}")
        return {"error": str(e)}



@app.get("/agent")
async def get_agent_thinking():
    """Get current agent thinking status and recent LLM interactions"""
    try:
        # Get the most recent LLM log file
        import glob
        import os
        from utils.llm_logger import get_llm_logger
        
        # Get recent LLM interactions
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()
        
                # Find all LLM log files and get interactions from all of them
        import glob
        log_files = glob.glob("llm_logs/llm_log_*.jsonl")
        logger.info(f"Found {len(log_files)} log files: {log_files}")
        
        # Get recent interactions from all log files
        recent_interactions = []
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get interactions from this file
                        for line in lines:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("type") == "interaction":
                                    recent_interactions.append({
                                        "type": entry.get("interaction_type", "unknown"),
                                        "prompt": entry.get("prompt", ""),
                                        "response": entry.get("response", ""),
                                        "duration": entry.get("duration", 0),
                                        "timestamp": entry.get("timestamp", "")
                                    })
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.error(f"Error reading LLM log {log_file}: {e}")
        
        # Sort by timestamp and keep only the last 3 interactions
        recent_interactions.sort(key=lambda x: x.get("timestamp", ""))
        recent_interactions = recent_interactions[-3:]
        logger.info(f"Found {len(recent_interactions)} recent interactions")
        
        # Format the agent thinking display
        if recent_interactions:
            current_thought = f"Recent LLM interactions:\n"
            for i, interaction in enumerate(reversed(recent_interactions)):
                current_thought += f"\n{i+1}. {interaction['type'].upper()} ({interaction['duration']:.2f}s)\n"
                current_thought += f"   Q: {interaction['prompt']}\n"
                current_thought += f"   A: {interaction['response']}\n"
        else:
            current_thought = "No recent LLM interactions. Agent is ready to process game state."
        
        with step_lock:
            current_step = agent_step_count  # Use agent step count instead of frame step count
        
        return {
            "status": "thinking",
            "current_thought": current_thought,
            "confidence": 0.85,
            "timestamp": time.time(),
            "llm_session": session_summary,
            "recent_interactions": recent_interactions,
            "current_step": current_step
        }
        
    except Exception as e:
        logger.error(f"Error in agent thinking: {e}")
        return {
            "status": "error",
            "current_thought": f"Error getting agent thinking: {str(e)}",
            "confidence": 0.0,
            "timestamp": time.time()
        }

@app.post("/agent_step")
async def update_agent_step():
    """Update the agent step count (called by agent.py)"""
    global agent_step_count
    
    with step_lock:
        agent_step_count += 1
    
    return {"status": "updated", "agent_step": agent_step_count}

@app.get("/llm_logs")
async def get_llm_logs():
    """Get recent LLM log entries"""
    try:
        from utils.llm_logger import get_llm_logger
        
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()
        
        # Get recent log entries
        recent_entries = []
        if os.path.exists(llm_logger.log_file):
            try:
                with open(llm_logger.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get the last 20 entries
                    for line in lines[-20:]:
                        try:
                            entry = json.loads(line.strip())
                            recent_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading LLM log: {e}")
        
        return {
            "session_summary": session_summary,
            "recent_entries": recent_entries,
            "log_file": llm_logger.log_file
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM logs: {e}")
        return {"error": str(e)}

# Milestone checking is now handled by the emulator

@app.get("/milestones")
async def get_milestones():
    """Get current milestones achieved based on persistent tracking"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Get milestones directly from emulator
        return env.get_milestones()
        
    except Exception as e:
        logger.error(f"Error getting milestones: {e}")
        # Fallback to basic milestones if memory reading fails
        basic_milestones = [
            {"id": 1, "name": "GAME_STARTED", "category": "basic", "completed": True, "timestamp": time.time()},
            {"id": 2, "name": "EMULATOR_RUNNING", "category": "basic", "completed": True, "timestamp": time.time()},
        ]
        return {
            "milestones": basic_milestones,
            "completed": 2,
            "total": 2,
            "progress": 1.0,
            "tracking_system": "fallback",
            "error": str(e)
        }

# Global list to track recent button presses
recent_button_presses = []

@app.get("/recent_actions")
async def get_recent_actions():
    """Get recently pressed buttons"""
    global recent_button_presses
    return {
        "recent_buttons": recent_button_presses[-10:],  # Last 10 button presses
        "timestamp": time.time()
    }

@app.get("/debug/milestones")
async def debug_milestones():
    """Debug milestone tracking system"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Get current working directory and list milestone files
        import glob
        current_dir = os.getcwd()
        milestone_files = glob.glob("*milestones*.json")
        
        # Check if default milestone file exists and get its info
        default_file_info = None
        if os.path.exists(env.milestone_tracker.filename):
            try:
                with open(env.milestone_tracker.filename, 'r') as f:
                    default_data = json.load(f)
                default_file_info = {
                    "exists": True,
                    "size": os.path.getsize(env.milestone_tracker.filename),
                    "last_modified": time.ctime(os.path.getmtime(env.milestone_tracker.filename)),
                    "milestone_count": len(default_data.get('milestones', {})),
                    "last_updated": default_data.get('last_updated', 'unknown')
                }
            except Exception as e:
                default_file_info = {"exists": True, "error": str(e)}
        else:
            default_file_info = {"exists": False}
        
        return {
            "tracking_system": "file_based",
            "current_filename": env.milestone_tracker.filename,
            "current_milestones": len(env.milestone_tracker.milestones),
            "completed_milestones": sum(1 for m in env.milestone_tracker.milestones.values() if m.get("completed", False)),
            "default_file_info": default_file_info,
            "milestone_files_in_directory": milestone_files,
            "working_directory": current_dir,
            "milestone_details": env.milestone_tracker.milestones
        }
    except Exception as e:
        logger.error(f"Error in milestone debug: {e}")
        return {"error": str(e)}

@app.post("/debug/reset_milestones")
async def reset_milestones():
    """Reset all milestones (for testing)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        env.milestone_tracker.reset_all()
        return {
            "status": "reset",
            "milestone_file": env.milestone_tracker.filename,
            "remaining_milestones": len(env.milestone_tracker.milestones)
        }
    except Exception as e:
        logger.error(f"Error resetting milestones: {e}")
        return {"error": str(e)}

@app.post("/debug/test_milestone_operations")
async def test_milestone_operations():
    """Test milestone loading and saving operations"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Test data
        test_milestones = {
            "TEST_MILESTONE_1": {
                "completed": True,
                "timestamp": time.time(),
                "first_completed": time.time()
            },
            "TEST_MILESTONE_2": {
                "completed": False,
                "timestamp": None
            }
        }
        
        # Save current state
        original_milestones = env.milestone_tracker.milestones.copy()
        original_filename = env.milestone_tracker.filename
        
        # Test 1: Save milestones with state filename
        test_state_filename = "test_state_123.sav"
        env.milestone_tracker.milestones = test_milestones.copy()
        saved_filename = env.milestone_tracker.save_milestones_for_state(test_state_filename)
        
        # Test 2: Load milestones for state
        env.milestone_tracker.milestones = {}  # Clear current milestones
        env.milestone_tracker.load_milestones_for_state(test_state_filename)
        loaded_milestones = env.milestone_tracker.milestones.copy()
        
        # Test 3: Check if file was created
        file_exists = os.path.exists(saved_filename)
        file_size = os.path.getsize(saved_filename) if file_exists else 0
        
        # Restore original state
        env.milestone_tracker.milestones = original_milestones
        env.milestone_tracker.filename = original_filename
        
        return {
            "test_results": {
                "save_operation": {
                    "filename": saved_filename,
                    "file_exists": file_exists,
                    "file_size": file_size,
                    "milestones_saved": len(test_milestones)
                },
                "load_operation": {
                    "milestones_loaded": len(loaded_milestones),
                    "milestones_match": loaded_milestones == test_milestones,
                    "loaded_milestones": loaded_milestones
                }
            },
            "original_state_restored": True
        }
        
    except Exception as e:
        logger.error(f"Error testing milestone operations: {e}")
        return {"error": str(e)}

@app.post("/stop")
async def stop_server():
    """Stop the server"""
    global running
    running = False
    return {"status": "stopping"}

def main():
    """Main function"""
    import argparse
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Simple Pokemon Emerald Server")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--manual", action="store_true", help="Enable manual mode with keyboard input and overlay")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    
    args = parser.parse_args()
    
    print("Starting Fixed Simple Pokemon Emerald Server")
    if args.manual:
        print("Manual mode enabled - keyboard input and overlay active")
    else:
        print("Server mode - no keyboard input, no overlay")
    print("Press Ctrl+C to stop")
    
    # Initialize pygame
    init_pygame()
    
    # Initialize emulator
    if not setup_environment():
        print("Failed to initialize emulator")
        return
    
    # Load state if specified
    if args.load_state:
        try:
            env.load_state(args.load_state)
            print(f"Loaded state from: {args.load_state}")
            
            # Map buffer should already be found by emulator.load_state()
            if env.memory_reader and env.memory_reader._map_buffer_addr:
                print(f"Map buffer already initialized at 0x{env.memory_reader._map_buffer_addr:08X}")
        except Exception as e:
            print(f"Failed to load state from {args.load_state}: {e}")
            print("Continuing with fresh game state...")
    
    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=run_fastapi_server, args=(args.port,), daemon=True)
    server_thread.start()
    
    print(f"FastAPI server running on http://127.0.0.1:{args.port}")
    print("Available endpoints:")
    print("  /status - Server status")
    print("  /screenshot - Current screenshot")
    print("  /action - Take action (POST)")
    print("  /state - Comprehensive game state (visual + memory data)")
    print("  /agent - Agent thinking status")
    print("  /milestones - Current milestones achieved")
    print("  /recent_actions - Recently pressed buttons")
    print("  /debug/memory - Debug memory reading (basic)")
    print("  /debug/memory/comprehensive - Comprehensive memory diagnostics")
    print("  /debug/milestones - Debug milestone tracking system")
    print("  /debug/reset_milestones - Reset all milestones (POST)")
    print("  /debug/test_milestone_operations - Test milestone save/load (POST)")
    print("  /stop - Stop server")
    
    try:
        # Run pygame loop in main thread
        game_loop(manual_mode=args.manual)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        global running
        running = False
        if env:
            env.stop()
        pygame.quit()
        print("Server stopped")

if __name__ == "__main__":
    main() 