import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from pynput import keyboard, mouse
from numba import njit

def main() -> None:
    """
        Run the main program.
    """

    global key
    key = None
    listener = set_up_input_systems()

    map_size = 15
    player_position, player_rotation = make_player(map_size)
    light_position = place_light(map_size)
    map_data, exit_position = build_maze(player_position, map_size)

    detail_level = 1
    ray_length, screen_size = set_up_resolution(detail_level)
    aspect = screen_size[1] / screen_size[0]
    color_buffer_size = (screen_size[1], screen_size[0], 3)
    window_surface = set_up_display(color_buffer_size)
    color_buffer = np.zeros(color_buffer_size)

    while True:

        forwards, right, up = get_camera_vectors(player_rotation)

        #Render
        for pixel_y in range(screen_size[1]):
            vertical_coeff = 2.0 * (pixel_y / screen_size[1]) - 1.0
            for pixel_x in range(screen_size[0]):
                horizontal_coeff = 2.0 * (pixel_x / screen_size[0]) - 1.0

                ray_position = player_position
                ray_direction = forwards \
                    - horizontal_coeff * right \
                    - aspect * vertical_coeff * up
                ray_direction = ray_length * ray_direction \
                    / np.linalg.norm(ray_direction)
                
                ray_position, color, distance_to_light = \
                    view_ray(ray_position, ray_direction, map_data, 
                             light_position, exit_position)
                
                x, y, z = ray_position
                if z < 1:
                    color = shadow_ray(ray_position, map_data, 
                                       light_position, color, distance_to_light)
                    #check reflectance
                    if map_data[2][int(x)][int(y)] != 0 and z > 0:
                        color = reflection(ray_position, ray_direction, 
                                           map_data, light_position, exit_position, 
                                           color, depth=0)
                color_buffer[pixel_y][pixel_x][:] = color[:]
        
        #gamma correction
        color_buffer = np.sqrt(color_buffer)
        window_surface.set_array(color_buffer); plt.draw(); plt.pause(0.0001)
        
        player_rotation = update_rotation(player_rotation)
        player_position, player_rotation = \
            update_movement(player_position, player_rotation, map_data[1], key)
        should_quit = key == keyboard.Key.esc
        key = None

        if on_position(player_position, exit_position) or should_quit:
            break        
            
    shutdown(listener)

#---- Program Setup and Teardown ----#
#region
def set_up_input_systems() -> keyboard.Listener:
    """
        Set up the input systems and return any input objects
        which must be managed.
    """

    listener = keyboard.Listener(on_press = on_press)
    listener.start()
    return listener

def make_player(size: int) -> tuple[tuple[float], tuple[float]]:
    """
        Initialize the player.

        Parameters:

            size: size of the map (tiles)

        Returns:

            The position and orientation of the player.
            eg. position, rotation = make_player(map_size)
    """
    position = (1, np.random.randint(1, size -1), 0.5)
    rotation = (np.pi/4, 0)
    return position, rotation

def place_light(map_size: int) -> tuple[float]:
    """
        Place a light! Returns the position of the placed
        light.
    """
    return (map_size/2-0.5, map_size/2-0.5, 1)

def build_maze(player_position: tuple[float], 
               map_size: int) -> tuple[tuple[np.ndarray], tuple[int]]:
    """
        Build a maze for the game.

        Parameters:

            player_position: the (x,y,z) position of the player

            map_size: the size of the map
        
        Returns:

            all of the map data (colors, reflectance, height) as well
            as the position of the exit.

            eg. map_data, exit_position = build_maze(player_position, map_size)
    """

    #colors are completely random
    colors = np.random.uniform(0,1, (map_size,map_size,3))
    #reflectance is either 0 or 1
    reflectance = np.random.choice([0, 0, 0, 0, 1], (map_size, map_size))
    #height of each grid block
    heights = np.random.choice([0, 0, 0, 0, 0, 0, 0, .3, .4, .7, .9], 
                               (map_size, map_size))

    clear_grid_block(colors, heights, reflectance, player_position)
    
    #Random walk, starting at the player's position
    count = 0 
    map_x, map_y, _ = player_position
    while True:
        
        #perform a random step
        if np.random.uniform() > 0.5:
            map_x = min(map_size - 2, max(1, map_x + np.random.choice([-1, 1])))
        else:
            map_y = min(map_size - 2, max(1, map_y + np.random.choice([-1, 1])))
        map_position = (map_x, map_y)
        
        #clear out a block and possibly lay an exit.
        if heights[map_x][map_y] == 0 or count > 5:
            count = 0
            clear_grid_block(colors, heights, reflectance, map_position)
            if map_x == map_size-2:
                break
        else:
            count += 1

    build_boundaries(heights, map_size)

    return (colors, heights, reflectance), (map_position)

def build_boundaries(heights: np.ndarray, map_size: int) -> None:
    """
        Enforce the boundary heights.
    """
    heights[0,:] = 1
    heights[map_size-1,:] = 1
    heights[:,0] = 1
    heights[:,map_size-1] = 1

def clear_grid_block(colors: np.ndarray, heights: np.ndarray, 
             reflectance: np.ndarray, position: tuple[int]) -> None:
    """
        Clear a block on the map at the given position.

        Parameters:

            colors: color of each map block

            heights: height of each map block

            reflectance: reflectance of each map block

            position: position to clear.
    """

    x = position[0]
    y = position[1]
    colors[x][y] = (0, 0, 0)
    heights[x][y] = 0
    reflectance[x][y] = 0

def set_up_resolution(detail_level: int) -> tuple[float, tuple[int]]:
    """
        Calculate parameters relating to the display.

        Parameters:

            detail_level: indicates the resolution

        Returns:

            the ray speed and screen size
    """
    ray_speed = 0.05 / detail_level
    screen_size = (int(60*detail_level), int(45*detail_level))
    return ray_speed, screen_size

def set_up_display(color_buffer_size: tuple[int]) -> image.AxesImage:
    """
        Set up the display and image for rendering.

        Parameters:

            screen_size: the size of the screen
        
        Returns:

            the window surface which will be drawn to
    """
    
    window = plt.figure().gca()
    window_surface = window.imshow(np.random.rand(*color_buffer_size))
    plt.axis('off'); plt.tight_layout()
    return window_surface

def shutdown(listener: keyboard.Listener) -> None:
    """
        Close any active resources.
    """

    plt.close()
    listener.stop()

#endregion
#---- Player Movement ----#
#region
def update_rotation(angles: tuple[float]) -> tuple[float]:
    """
        Update the player's angles based on the mouse's position.

        Parameters:

            angles: the player's current angles

            screen_size: size of the screen
        
        Returns:

            the player's angles after rotation
    """

    #unpack data
    theta, phi = angles
    #the input system is kind of janky and
    #measures the mouse's position, indpendent of the window.
    screen_width, screen_height = 1920,1080

    #update angles
    with mouse.Controller() as check:
        position = check.position

        dx = (screen_width / 2 - position[0])/4800
        if (abs(dx) > 0.0625):
            theta = np.real(theta - dx)

        dy = (screen_height / 2 - position[1])/2700
        if (abs(dy) > 0.0625):
            phi = np.real(phi + dy)
    
    #boundary conditions
    theta = np.mod(theta, 2 * np.pi)
    phi = np.clip(phi, -np.pi / 2 + 0.001, np.pi / 2 - 0.001)

    return (theta, phi)

def on_press(key_new: keyboard.KeyCode) -> None:
    """
        Keypress callback function.
    """
    global key
    key = key_new
    
def update_movement(player_position: tuple[float], 
                    player_rotation: tuple[float], heights: np.ndarray, 
                    key: keyboard.KeyCode) -> tuple[tuple[float], tuple[float]]:
    """
        Move the player.

        Parameters:

            player_position, player_rotation: the player's data

            heights: heights of each block on the map

            key: the most recent keypress

        Returns:

            the player's position and player's rotation
    """
    
    #fetch data
    x, y, z = player_position
    rot, rot_v = player_rotation

    if key is not None:
        if key == keyboard.Key.up:
            x, y = (x + 0.3*np.cos(rot), y + 0.3*np.sin(rot))
        elif key == keyboard.Key.down:
            x, y = (x - 0.3*np.cos(rot), y - 0.3*np.sin(rot))
        elif key == keyboard.Key.left:
            rot = rot - np.pi/8
        elif key == keyboard.Key.right:
            rot = rot + np.pi/8
        elif key == keyboard.Key.end:
            rot_v = rot_v - np.pi/16
        elif key == keyboard.Key.home:
            rot_v = rot_v + np.pi/16
    if rot > 2 * np.pi:
        rot -= 2 * np.pi
    if rot < 0:
        rot += 2 * np.pi

    if heights[int(x)][int(y)] == 0:
        player_position = (x, y, z)
        
    return player_position, (rot, rot_v)

def get_camera_vectors(angles: tuple[float]) -> tuple[np.ndarray]:
    """
        Get the camera's orthonormal direction vectors.

        Parameters:

            angles: the angles of rotation for the camera around the z and y axes.

        Returns:

            the camera's forwards, right and up vectors
    """

    c = np.cos(angles[0])
    s = np.sin(angles[0])
    c2 = np.cos(angles[1])
    s2 = np.sin(angles[1])
    forwards = np.array([c * c2, s * c2, s2])
    forwards = forwards / np.linalg.norm(forwards)
    global_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forwards, global_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forwards)
    up = up / np.linalg.norm(up)

    return forwards, right, up
#endregion
#---- Rays and Geometry ----#
#region
@njit(fastmath=True)
def shoot(ray_position: tuple[float], ray_direction: np.ndarray, 
          block_heights: np.ndarray) -> tuple[float]:
    """
        Step the given ray forwards until it hits something.

        Parameters:

            ray_position, ray_direction: describe the ray

            block_heights: heights of the blocks

        Returns:

            the final position of the ray after walking
    """
    #unpack data
    x,y,z = ray_position
    dx, dy, dz = ray_direction

    while True:
        #step forwards
        x, y, z = x + dx, y + dy, z + dz

        #did the ray hit the ceiling or floor?
        if (z >= 1 or z <= 0):
            break

        #did the ray hit a block?
        if block_heights[int(x)][int(y)] > z:
            break        
    return x, y, z        

def view_ray(ray_position: tuple[float], ray_direction: np.ndarray, 
             map_data: tuple[np.ndarray], light_position: tuple[int], 
             exit_position: tuple[int]) -> tuple[tuple[float], np.ndarray, float]:
    """
        Cast a primary ray out into the world.

        Parameters:

            ray_position, ray_direction: ray origin and direction

            map_data: colors, heights and reflectance of blocks in the world

            light_position, exit_position: additional info about
                the world
            
        Returns:

            the ray's intersection position, color of the intersected
            point, and the point's distance from the light.
    """

    block_colors, block_heights, _ = map_data

    ray_position = shoot(ray_position, ray_direction, block_heights)
    x,y,z = ray_position

    #Decide color for intersection position
    # this is a placeholder value/definition
    color = np.asarray([.5,.5,.5])
    #Case: Ceiling
    if z > 1:
        if distance(ray_position, light_position, 2) < np.sqrt(0.1):
            color = np.asarray([1,1,1])
        elif ceiling_pattern(ray_position, light_position):
            color = np.asarray([.6,1,1])
        else:
            color = np.asarray([1,1,0.6])
        
    #Case: Floor
    elif z < 0:
        if on_position((x,y), exit_position):
            color = np.asarray([0,0,.6])
        elif floor_pattern(x,y):
            color = np.asarray([.1,.1,.1])
        else:
            color = np.asarray([.8,.8,.8])

    #Case: Wall
    elif z < block_heights[int(x)][int(y)]:
        color = np.asarray(block_colors[int(x)][int(y)])

    #Apply lighting to point.
    distance_to_light = distance(ray_position, light_position, 3)
    h = 0.3 + 0.7*np.clip(1/distance_to_light, 0, 1)
    color = h * color

    return ray_position, color, distance_to_light

@njit(fastmath = True)
def shadow_ray(ray_position: tuple[float], map_data: tuple[np.ndarray], 
               light_position: tuple[int], color: np.ndarray,
               distance_to_light: float) -> np.ndarray:
    """
        Trace in the direction of the light and approximate the amount of shadow
        at the given intersection point.

        Parameters:

            ray_position: position to apply lighting to

            map_data: holds information about the world

            light_position: position of the light

            color: current color of the position to light

            distance_to_light: For normalization purposes
    """
    #unpack data
    block_heights = map_data[1]
    x,y,z = ray_position
    lx, ly, lz = light_position

    #unit vector pointing towards light
    step_size = 0.1
    dx = step_size * (lx-x)/distance_to_light
    dy = step_size * (ly-y)/distance_to_light
    dz = step_size * (lz-z)/distance_to_light
    intensity = 1
    """
        The update scheme is strange, here's how it works:

            step towards the light

            are we currently in a block? Reduce the light energy

            stop once we're close enough to the light, or have lost
            a certain amount of energy.

        The upshot is that this method can give a decent
        approximation of soft shadows.
    """
    while True:
        x, y, z = x + dx, y + dy, z + dz
        distance_to_light = distance_to_light - step_size
        if z <= block_heights[int(x)][int(y)]:
            intensity = intensity*0.9
            if intensity < 0.5:
                break
        elif z > 0.9 or distance_to_light <= 0:
            break
    return color * intensity

def reflection(ray_position: tuple[float], ray_direction: np.ndarray, 
               map_data: tuple[np.ndarray], 
               light_position: tuple[float], exit_position: tuple[int], 
               color: np.ndarray, depth: int) -> np.ndarray:
    """
        Shoot a reflection ray and incorporate its result into the
        ray's current color.

        Parameters:

            ray_position, ray_direction: describe the ray

            map_data: describe the world

            light_position, exit_position: extra world info

            color: the ray's current color

            depth: how many bounces have already been performed (used for
                recursion limiting)

        Returns:

            the resulting color
    """
    
    #unpack data
    _, block_heights, block_reflectances = map_data
    x,y,z = ray_position

    #approximate reflection
    if abs(z-block_heights[int(x)][int(y)])<abs(ray_direction[2]):
        ray_direction[2] *= -1
    elif block_heights[int(x+ray_direction[0])][int(y-ray_direction[1])] != 0:
        ray_direction[0] *= -1
    else:
        ray_direction[1] *= -1
    
    #trace again!
    ray_position, new_color, distance_to_light = view_ray(
        ray_position, ray_direction, 
        map_data, light_position, exit_position)
    x,y,z = ray_position
    if z < 1:
        new_color = shadow_ray(
            ray_position, map_data, 
            light_position, new_color, distance_to_light)
    max_depth = 1
    if (block_reflectances[int(x)][int(y)] != 0 
        and z < 1 and z > 0 
        and depth < max_depth):
        new_color = reflection(ray_position, ray_direction, 
                               map_data, light_position, exit_position, 
                               new_color, depth + 1)
    color = (color + new_color) / 2
    return color

@njit(fastmath=True)
def on_position(pos_a: tuple[float], pos_b: tuple[int]) -> bool:
    """
        Returns whether pos_a (approximate) is on top of pos_b (exact)
    """

    return int(pos_a[0]) == pos_b[0] and int(pos_a[1]) == pos_b[1]

def distance(pos_a: tuple[float], pos_b: tuple[float], dimension: int) -> float:
    """
        Return the euclidean distance between two n-dimensional points.
    """

    d = 0.0
    for i in range(dimension):
        d += (pos_a[i] - pos_b[i]) ** 2
    return np.sqrt(d)

@njit(fastmath = True)
def ceiling_pattern(ray_position: tuple[float], 
                      light_position: tuple[float]) -> bool:
    """
        Apply a radial pattern based on two positions on the ceiling.
    """
    dx = ray_position[0] - light_position[0]
    dy = ray_position[1] - light_position[1]
    theta = np.rad2deg(np.arctan(dy / dx))
    return int(theta / 6) % 2 == 1

@njit(fastmath = True)
def floor_pattern(x: float, y: float) -> bool:
    """
        Generator for a floor pattern.
    """
    return int(x*2)%2 == int(y*2)%2
#endregion
if __name__ == '__main__':
    main()
