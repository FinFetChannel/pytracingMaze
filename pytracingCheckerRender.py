import numpy as np
import pygame as pg
from numba import njit

def main():
    size = np.random.randint(20,60) # size of the map
    posx, posy, posz = 1.5, np.random.randint(1, size -2)+0.5, 0.5
    
    rot, rot_v = (np.pi/4, 0)
    lx, ly, lz = (size*20, size*30, 1000)
    mr, mg, mb, maph, mapr, exitx, exity, mapt, maps = maze_generator(int(posx), int(posy), size)
    enx, eny, seenx, seeny, lock  = np.random.uniform(2, size-3 ), np.random.uniform(2, size-3), 0, 0, 0
    maph[int(enx)][int(eny)] = 0
    shoot, sx, sy, sdir = 1, -1, -1, rot
    shoot2, sx2, sy2, sdir2 = 1, -1, -1, rot

    width, height, mod, inc, rr, gg, bb = adjust_resol(24)

    running, pause, fps_lock = 1, 0, 60
    pg.init()
    
    font = pg.font.SysFont("Arial", 18)
    font2 = pg.font.SysFont("Impact", 48)
    screen = pg.display.set_mode((800, 600))
    rr, gg, bb = np.linspace(0,0.8, width*height), np.linspace(0.5,.1, width*height), np.linspace(1,0.1, width*height)
    pixels = np.dstack((rr,gg,bb))
    pixels = np.reshape(pixels, (height,width,3))/np.max(pixels)
    surf = pg.surfarray.make_surface((np.rot90(pixels*255)).astype('uint8'))
    surf = pg.transform.scale(surf, (750, 550))
    screen.blit(surf, (25, 25))
    endmsg = " FinFET's PyTracing Maze  "
    screen.blit(font2.render(endmsg, 1, pg.Color("red")),(45,95))
    screen.blit(font2.render(endmsg, 1, pg.Color("blue")),(55,105))
    screen.blit(font2.render(endmsg, 1, pg.Color("white")),(50,100))
    screen.blit(font2.render(" Loading, please wait... ", 1, pg.Color("black"), pg.Color("grey")),(50,300))
    pg.display.update()
    pg.time.wait(200)
    clock = pg.time.Clock()
    pg.mouse.set_visible(False)
    et = 0.1
    mplayer = np.zeros([size, size])
    enx, eny, mplayer, et, shoot, sx, sy, sdir, shoot2, sx2, sy2, sdir2, seenx, seeny, lock = agents(enx, eny, maph, posx, posy, rot, et, shoot, sx, sy, sdir,
                                                                                                     shoot2, sx2, sy2, sdir2, mplayer, seenx, seeny, lock, size)
    sstart, sstart2, timer, count, autores, smooth, checker = None, None, 0, -100, 1, 0, 1
    

    surfbg = pg.Surface((800,600))
    surfmap = pg.Surface((size,size))
    minimap = np.zeros((size, size, 3))
    pg.mixer.set_num_channels(4)
    ambient = pg.mixer.Sound('soundfx/HauntSilentPartner.mp3')
    ambient.set_volume(0.5)
    runfx = pg.mixer.Sound('soundfx/run.mp3')
    shotfx = pg.mixer.Sound('soundfx/slap.mp3')
    killfx = pg.mixer.Sound('soundfx/shutdown.mp3')
    respawnfx = pg.mixer.Sound('soundfx/respawn.mp3')
    successfx = pg.mixer.Sound('soundfx/success.mp3')
    failfx = pg.mixer.Sound('soundfx/fail.mp3')
    pg.mixer.Channel(0).play(ambient, -1)
    pg.mixer.Channel(1).play(respawnfx)
    run, respawn = 1, 0
    score, health = 0, 10
    ticks = pg.time.get_ticks()/100000
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                if not pause:
                    pause = 1
                    pg.mixer.Channel(1).play(respawnfx)
                    endmsg = " Game paused. Current score: " + str(score) + ' '
                else:
                    endmsg = " Thanks for playing! Total score: " + str(score) + ' '
                    pg.mixer.Channel(1).play(killfx)
                    running = False
            if sstart == None and(event.type == pg.MOUSEBUTTONDOWN or event.type == pg.MOUSEBUTTONUP):
                shoot = 1
            if event.type == pg.KEYDOWN:
                if event.key == ord('p'): # pause
                    if not pause:
                        pause = 1
                        endmsg = " Game paused. Current score: " + str(score)
                    elif (int(posx) != exitx or int(posy) != exity):
                        pause = 0
                if pause and event.key == ord('n'): # new game
                    pause = 0
                    size = np.random.randint(20,60)
                    posx, posy, posz = 1.5, np.random.randint(1, size -2)+0.5, 0.5
                    rot, rot_v = (np.pi/4, 0)
                    mr, mg, mb, maph, mapr, exitx, exity, mapt, maps = maze_generator(int(posx), int(posy), size)
                    enx, eny, seenx, seeny, lock, run  = 0, 0, 0, 0, 0, 1
                    shoot, sx, sy, sstart = 0, -1, -1, None
                    mplayer = np.zeros([size, size])
                    et = 0.1
                    enx, eny, mplayer, et, shoot, sx, sy, sdir, shoot2, sx2, sy2, sdir2, seenx, seeny, lock = agents(enx, eny, maph, posx, posy, rot, et, shoot,
                                                                                                                     sx, sy, sdir, shoot2, sx2, sy2, sdir2, mplayer,
                                                                                                                     seenx, seeny, lock, size)
                    count, health = -100, 10
                    minimap = np.zeros((size, size, 3))
                    if autores:
                        width, height, mod, inc, rr, gg, bb = adjust_resol(24)
                    pg.mixer.Channel(1).play(respawnfx)
                if event.key == ord('t'): # toggle auto resolution
                    autores = not(autores)
                if event.key == ord('y'): # toggle smooth scaling
                    smooth = not(smooth)
                if event.key == ord('u'): # toggle checkerboard rendering
                    checker = not(checker)

                if event.key == ord('q'): # manually change resolution down
                    if autores:
                        fps_lock = max(20, fps_lock - 10)
                    else:
                        if width > 100 :
                            count = 0
                            width, height, mod, inc, rr, gg, bb = adjust_resol(int(width*0.8))
                if event.key == ord('e'): # manually change resolution up
                    if autores:
                        fps_lock = min(120, fps_lock + 10)
                    else:
                        count = 0
                        width, height, mod, inc, rr, gg, bb = adjust_resol(int(width*1.1))
            
        if not pause:
            ticks = pg.time.get_ticks()/100000
            lx, ly = size/2 + 1500*np.cos(ticks), size/2 + 1000*np.sin(ticks)
            surfbg.fill(pg.Color("deepskyblue3"))
            rr, gg, bb = super_fast(width, height, mod, inc, posx, posy, posz, rot, rot_v, mr, mg, mb, lx, ly, lz,
                                    mplayer, exitx, exity, mapr, mapt, maps, rr, gg, bb, enx, eny, sx, sy, sx2, sy2, size, checker, count)
            count += 1
            pixels = np.dstack((rr,gg,bb))

            pixels = np.reshape(pixels, (height,width,3))

            surf = pg.surfarray.make_surface((np.rot90(pixels*255)).astype('uint8'))
            
            if smooth:
                surf = pg.transform.smoothscale(surf, (800, 550))
            else:
                surf = pg.transform.scale(surf, (800, 550))
                
            pg.draw.rect(surfbg, pg.Color("forestgreen"),(0,579,800,19))
            if health < 10:
                pg.draw.rect(surfbg, pg.Color("firebrick"),(int(400-40*(10-health)),581,int(80*(10-health)),15))
            screen.blit(surfbg, (0, 0))
            screen.blit(surf, (0, 25))

            minimap[int(posy)][int(posx)] = (50, 50, 255)
            surfmap = pg.surfarray.make_surface(np.flip(minimap).astype('uint8'))
            minimap[int(posy)][int(posx)] = (100, 100, 0)
            surfmap = pg.transform.scale(surfmap, (size*3, size*3))
            screen.blit(surfmap,(800-size*3, 25), special_flags=pg.BLEND_ADD)
            
            if enx != 0 and lock:
                endmsg = 'Pytracing Maze -   Watch out!      Score:'
            else:
                endmsg = 'Pytracing Maze - Find the exit!    Score:'
            fps = font.render(endmsg + str(score)+' Res: '+ str(width) +'x' + str(height) +
                              '  FPS: '+ str(int(clock.get_fps())), 1, pg.Color("coral"))
            screen.blit(fps,(100,1))
            fpss = int(1000/(pg.time.get_ticks() - ticks*100000))

            if autores and count > 10: #auto adjust render resolution
                if fpss < fps_lock - 10 and width > 100:
                        count, garbage = 0, 1
                        width, height, mod, inc, rr, gg, bb = adjust_resol(int(width*0.8))
                if fpss > fps_lock + 15:
                        count, garbage = 0, 1
                        width, height, mod, inc, rr, gg, bb = adjust_resol(int(width*1.1))        

            if (int(posx) == exitx and int(posy) == exity):
                endmsg = " You escaped safely! "
                pg.mixer.Channel(1).play(successfx)
                score += 1
                pause = 1
            
            pressed_keys = pg.key.get_pressed()
            et = clock.tick()/500
            if et > 0.5:
                et = 0.5

            if shoot or sstart != None:
                if sstart == None:
                    pg.mixer.Channel(2).play(shotfx)
                    if fpss < fps_lock and autores:
                        count, garbage = 0, 1
                        width, height, mod, inc, rr, gg, bb = adjust_resol(int(width*0.8))
                    sstart = pg.time.get_ticks()
                elif pg.time.get_ticks() - sstart > 500:
                    shoot, sx, sy, sstart = 0, -1, -1, None

            if enx == 0:
                respawn = 1
                if not run:
                    pg.mixer.Channel(1).play(killfx)
                    run = 1
                    if shoot:
                        health += 1
            else:
                if respawn:
                    respawn = 0
                    pg.mixer.Channel(1).play(respawnfx)    
                if shoot2 or sstart2 != None:
                    if sstart2 == None:
                        pg.mixer.Channel(3).play(shotfx)
                        sstart2 = pg.time.get_ticks()
                    elif pg.time.get_ticks() - sstart2 > 500:
                        shoot2, sx2, sy2, sstart2 = 0, -1, -1, None
                        
                    if (sx2 - posx)**2 + (sy2 - posy)**2 < 0.01:
                        health -= 1 + score/2
                        if health < 0:
                            health = 5
                            if score > 0:
                                score -= 1
                            endmsg = " You died! Current score: " + str(score) + ' '
                            pg.mixer.Channel(1).play(failfx)
                            enx, eny, seenx, seeny, lock  = 0, 0, 0, 0, 0
                            shoot2, sx2, sy2, sstart2 = 0, -1, -1, None
                            pause = 1
                            surf = pg.surfarray.make_surface((np.rot90(255-pixels*255)).astype('uint8'))
                            surf = pg.transform.smoothscale(surf, (800, 550))
                            screen.blit(surf, (25, 0))

                if (enx-posx)**2 + (eny-posy)**2 > 300:
                    enx, eny, seenx, seeny, lock  = 0, 0, 0, 0, 0
                    run = 0

            posx, posy, rot, rot_v, shoot = movement(pressed_keys,posx, posy, rot, rot_v, maph, et, shoot, sstart)
            pg.mouse.set_pos([400, 300])
            mplayer = np.zeros([size, size])
            enx, eny, mplayer, et, shoot, sx, sy, sdir, shoot2, sx2, sy2, sdir2,seenx, seeny, lock = agents(enx, eny, maph, posx, posy, rot, et, shoot,
                                                                                                            sx, sy, sdir, shoot2, sx2, sy2, sdir2,
                                                                                                            mplayer, seenx, seeny, lock, size)
            if run and (seenx == posx or seeny == posy):
                run = 0
                pg.mixer.Channel(1).play(runfx)
        else:
            clock.tick(30)
            screen.blit(font2.render(" FinFET's PyTracing Maze  ", 1, pg.Color("red")),(45,45))
            screen.blit(font2.render(" FinFET's PyTracing Maze  ", 1, pg.Color("blue")),(55,55))
            screen.blit(font2.render(" FinFET's PyTracing Maze ", 1, pg.Color("white")),(50,50))
            screen.blit(font2.render(endmsg, 1, pg.Color("salmon"), (100, 34, 60)),(50,320))
            if (int(posx) == exitx and int(posy) == exity):
                screen.blit(font2.render(" Your current score is "+str(score) + ' ', 1, pg.Color("grey"), (80, 34, 80)),(50,390))
            else:
                screen.blit(font2.render(" Press P to continue ", 1, pg.Color("grey"), (80, 34, 80)),(50,390))
            screen.blit(font2.render(" Press N for a new game ", 1, pg.Color("grey"), (45, 34, 100)),(50,460))
            screen.blit(font2.render(" Press ESC to leave ", 1, pg.Color("grey"), (13, 34, 139)),(50,530))
        
        pg.display.update()

    screen.blit(font2.render(endmsg, 1, pg.Color("salmon"), (100, 34, 60)),(50,320))
    pg.mixer.fadeout(1000)
    pg.display.update()
    print(endmsg)
    pg.time.wait(1000)
    pg.quit()

def maze_generator(x, y, size):
    
    mr = np.random.uniform(0,1, (size,size)) 
    mg = np.random.uniform(0,1, (size,size)) 
    mb = np.random.uniform(0,1, (size,size)) 
    mapr = np.random.choice([0, 0, 0, 0, 1], (size,size))
    maps = np.random.choice([0, 0, 0, 0, 1], (size,size))
    mapt = np.random.choice([0, 0, 0, 1, 2], (size,size))
    maptemp = np.random.choice([0,0, 1], (size,size))
    maph = np.random.uniform(0.25, 0.99, (size,size))
    maph[np.where(maptemp == 0)] = 0
    maph[0,:], maph[size-1,:], maph[:,0], maph[:,size-1] = (1,1,1,1)
    maps[0,:], maps[size-1,:], maps[:,0], maps[:,size-1] = (0,0,0,0)

    maph[x][y], mapr[x][y] = (0, 0)
    count = 0 
    while 1:
        testx, testy = (x, y)
        if np.random.uniform() > 0.5:
            testx = testx + np.random.choice([-1, 1])
        else:
            testy = testy + np.random.choice([-1, 1])
        if testx > 0 and testx < size -1 and testy > 0 and testy < size -1:
            if maph[testx][testy] == 0 or count > 5:
                count = 0
                x, y = (testx, testy)
                maph[x][y], mapr[x][y] = (0, 0)
                if x == size-2:
                    exitx, exity = (x, y)
                    break
            else:
                count = count+1
    
    return mr, mg, mb, maph, mapr, exitx, exity, mapt, maps

def movement(pressed_keys,posx, posy, rot, rot_v, maph, et, shoot, sstart):
    
    x, y = (posx, posy)
    p_mouse = pg.mouse.get_pos()
    rot, rot_v = rot - np.clip((p_mouse[0]-400)/200, -0.2, .2), rot_v -(p_mouse[1]-300)/400
    rot_v = np.clip(rot_v, -1, 1)

    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        x, y = (x + et*np.cos(rot), y + et*np.sin(rot))
        
    if pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        x, y = (x - et*np.cos(rot), y - et*np.sin(rot))
        
    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        x, y = (x - et*np.sin(rot), y + et*np.cos(rot))
        
    if pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        x, y = (x + et*np.sin(rot), y - et*np.cos(rot))
        
    if maph[int(x)][int(y)] == 0 and maph[int(x+0.05)][int(y)] == 0 and maph[int(x)][int(y+0.05)] == 0:
        posx, posy = (x, y)
    elif maph[int(posx)][int(y)] == 0 and maph[int(posx+0.05)][int(y)] == 0 and maph[int(posx)][int(y+0.05)] == 0:
        posy = y
    elif maph[int(x)][int(posy)] == 0 and maph[int(x+0.05)][int(posy)] == 0 and maph[int(x)][int(posy+0.05)] == 0:
        posx = x
        
    if not shoot and sstart == None and pressed_keys[pg.K_SPACE]:
        shoot = 1
                                                
    return posx, posy, rot, rot_v, shoot
        
@njit(cache=True)
def super_fast(width, height, mod, inc, posx, posy, posz, rot, rot_v, mr, mg, mb, lx, ly, lz,
               maph, exitx, exity, mapr, mapt, maps, pr, pg, pb, enx, eny, sx, sy, sx2, sy2, size, checker, count):
        
    texture=[[ .95,  .99,  .97, .8], # brick wall
             [ .97,  .95,  .96, .85],
             [.8, .85, .8, .8],
             [ .93, .8,  .98,  .96],
             [ .99, .8,  .97,  .95],
             [.8, .85, .8, .8]]

    inverse = count%2
    rot, rot_v = rot + inc*(1-inverse)/4, rot_v + inc*(1-inverse)/4
    garbage = not(count)
    idx = 0
    for j in range(height): #vertical loop 
        rot_j = rot_v + np.deg2rad(24 - j/mod)
        sinzo = inc*np.sin(rot_j) 
        coszo = inc*np.sqrt(abs(np.cos(rot_j)))        
        for i in range(width): #horizontal vision loop
            if (not(checker) or i == 0 or i == width -1 or j == 0 or j == height -1 or (inverse and i%2 == j%2) or (not(inverse) and i%2 != j%2)):
                rot_i = rot + np.deg2rad(i/mod - 30)
                x, y, z = (posx, posy, posz)
                sin, cos, sinz = coszo*np.sin(rot_i), coszo*np.cos(rot_i), sinzo
                   
                modr = 1
                cx, cy, c1r, c2r, c3r = 1, 1, 1, 1, 1
                shot, enem, mapv = 0, 0, 0
                dtp = np.random.uniform(0.002,0.01)
                while 1:
                    if (mapv == 0 or (sinz > 0 and (z > mapv or (mapv==6 and (z>0.4 or z <0.2)) or(z > 0.57 and mapv > 1)))): ## LoDev DDA for optimization
                        
                        norm = np.sqrt(cos**2 + sin**2 + sinz**2)
                        rayDirX, rayDirY, rayDirZ = cos/norm + 1e-16, sin/norm + 1e-16, sinz/norm + 1e-16
                        
                        mapX, mapY = int(x), int(y)

                        deltaDistX, deltaDistY, deltaDistZ= abs(1/rayDirX), abs(1/rayDirY), abs(1/rayDirZ)

                        if (rayDirX < 0):
                            stepX, sideDistX = -1, (x - mapX) * deltaDistX
                        else:
                            stepX, sideDistX = 1, (mapX + 1.0 - x) * deltaDistX
                            
                        if (rayDirY < 0):
                            stepY, sideDistY = -1, (y - mapY) * deltaDistY
                        else:
                            stepY, sideDistY = 1, (mapY + 1 - y) * deltaDistY

                        if (rayDirZ < 0):
                            sideDistZ = z*deltaDistZ;
                        else:
                            sideDistZ = (1-z)*deltaDistZ

                        while (1):
                            if (sideDistX < sideDistY):
                                sideDistX += deltaDistX; mapX += stepX
                                dist = sideDistX; side = 0
                                if mapX < 2 or mapX > size-2:
                                    break
                            else:
                                sideDistY += deltaDistY; mapY += stepY
                                dist = sideDistY; side = 1
                                if mapY < 2 or mapY > size-2:
                                    break
                            if (maph[mapX][mapY] != 0):
                                break
                                
                        if (side):
                            dist = dist - deltaDistY
                        else:
                            dist = dist - deltaDistX
                            
                        if (dist > sideDistZ):
                            dist = sideDistZ

                        x = x + rayDirX*dist - cos/2
                        y = y + rayDirY*dist - sin/2
                        z = z + rayDirZ*dist - sinz/2
                        
                        ## end of LoDev DDA
                    
                    x += cos; y += sin; z += sinz
                    if (z > 1 or z < 0): # check ceiling and floor
                        break
                    mapv = maph[int(x)][int(y)]
                    if mapv > 1 and z < 0.57:
                        if  mapv == 2 or mapv == 8:
                            if z> 0.45 and (x-posx)**2 + (y-posy)**2 + (z-0.5)**2 < 0.005 :
                                break
                            if z < 0.45 and z > 0.3 and (x-posx)**2 + (y-posy)**2  < (z/10 - 0.02):
                                break
                            if z < 0.3 and (x-posx)**2 + (y-posy)**2 + (z-0.15)**2 < 0.023 :
                                    break
                        if  mapv == 3 or mapv == 9:
                            enem = 1
                            if z> 0.45 and (x-enx)**2 + (y-eny)**2 + (z-0.5)**2 < 0.005 :
                                break
                            if z < 0.45 and z > 0.3 and (x-enx)**2 + (y-eny)**2  < (z/10 - 0.02):
                                break
                            if z < 0.3 and (x-enx)**2 + (y-eny)**2 + (z-0.15)**2 < 0.023 :
                                break
                        if  mapv > 5 and z < 0.4 and z > 0.2:
                            if mapv < 12:
                                if ((x-sx2)**2 + (y-sy2)**2 + (z-0.3)**2 < dtp):#0.01):
                                    shot = 1
                                    break
                            else:
                                if ((x-sx)**2 + (y-sy)**2 + (z-0.3)**2 < dtp):#0.01):
                                    shot = 1
                                    break

                    if mapv > z and mapv < 2: # check walls
                        if maps[int(x)][int(y)]: # check spheres
                            if ((x-int(x)-0.5)**2 + (y-int(y)-0.5)**2 + (z-int(z)-0.5)**2 < 0.25):
                                if (mapr[int(x)][int(y)]): # spherical mirror
                                    if (modr == 1):
                                        cx, cy = int(x), int(y)
                                    modr = modr*0.7
                                    if (modr < 0.2):
                                        break
                                    if (mapv - z <= abs(sinz) ): ## horizontal surface
                                        sinz = -sinz
                                    else:
                                        nx = (x-int(x)-0.5)/0.5; ny = (y-int(y)-0.5)/0.5; nz =(z-int(z)-0.5)/0.5
                                        dot = 2*(cos*nx + sin*ny + sinz*nz)
                                        cos = (cos - nx*dot); sin = (sin - ny*dot); sinz = (sinz - nz*dot)                
                                        x += cos; y += sin; z += sinz
                                else:
                                    break
                                        
                                    
                        elif mapr[int(x)][int(y)]: # check reflections
                            if modr == 1:
                                cx, cy = int(x), int(y)
                            modr  = modr*0.7
                            if modr < 0.2:
                                break
                            if abs(z-maph[int(x)][int(y)]) < abs(sinz):
                                sinz = -sinz
                            elif maph[int(x+cos)][int(y-sin)] == maph[int(x)][int(y)]:
                                cos = -cos
                            else:
                                sin = -sin
                        else:
                            break

                    
                if z > 1: # ceiling
                    deltaDistZ = (lz-z)*deltaDistZ
                    x += deltaDistZ*rayDirX; y += deltaDistZ*rayDirY; z = lz
                    dtol = np.sqrt((x-lx)**2+(y-ly)**2)

                    if dtol < 50: #light source
                        shot = 1
                        c1, c2, c3 = 1, 1, 0.5
                    else:
                        angle = np.rad2deg(np.arctan((y-ly)/(x-lx)))/np.random.uniform(12,15)
                        sh = (0.8+ abs(angle - int(angle))/5)/(dtol/1000)
                        if sh > 1:
                            sh = 1
                        if int(angle)%2 == 1:
                            c1, c2, c3 = 0.8*(1-sh), 0.86*(1-sh/4), (1-sh/10)
                        else:
                            c1, c2, c3 = 0.8*(1-sh), 0.9*(1-sh/4), (1-sh/10)
                    if sx != -1:
                        c1, c2, c3 = 0.7*c1, 0.7*c2, 0.7*c3
                        
                elif z < 0: # floor
                    z = 0
                    if int(x*2)%2 == int(y*2)%2:
                        c1, c2, c3 = .8,.8,.8
                    else:
                        if int(x) == exitx and int(y) == exity: #exit
                            c1, c2, c3 = 0,0,.6
                        else:
                            c1, c2, c3 = .1,.1,.1
                            
                elif mapv < 2: # walls
                    c1, c2, c3 = mr[int(x)][int(y)], mg[int(x)][int(y)], mg[int(x)][int(y)]
                    if mapt[int(x)][int(y)]: # textured walls
                        if y%1 < 0.05 or y%1 > 0.95:
                            ww = int((x*3)%1*4)
                        else:
                            ww = int((y*3)%1*4)
                        if x%1 < 0.95 and x%1 > 0.05 and y%1 < 0.95 and y%1 > 0.05:
                            zz = int(x*5%1*6)
                        else:
                            zz = int(z*5%1*6)
                        text = texture[zz][ww]
                        c1, c2, c3 = c1*text, c2*text, c3*text

                    if mapv - z <= abs(sinz):
                        z = mapv
                    elif not maps[int(x)][int(y)]:
                        if int(x-cos) != int(x):
                            x = max(int(x-cos), int(x))
                            modr = modr*0.80
                        else:
                            y = max(int(y-sin), int(y))
                            modr = modr*0.9
                else:
                    if shot:
                        if mapv < 12:
                            sh = ((x-sx2)**2 + (y-sy2)**2 + (z-0.3)**2)/0.012
                            c1, c2, c3 = 0.8, 0.5*sh+0.2 , 0.1*sh+0.1 # shot enemy
                        else:
                            sh = ((x-sx)**2 + (y-sy)**2 + (z-0.3)**2)/0.012
                            c1, c2, c3 = 1, 0.6*sh+0.2 , 0.2*sh+0.1 # shot
                    elif z> 0.45:
                        c1, c2, c3 = 0.6, 0.3, 0.3 # Head
                    elif z > 0.3:
                        c1, c2, c3 = 0.3, 0.5, 0.5 # Chest
                    else:
                        if enem:
                            c1, c2, c3 = 1, 0.2, 0.2 # Roller red
                        else:
                            c1, c2, c3 = 0.2, 0.2, 1 # Roller blue

                if modr <= 0.7 and not shot:
                    c1r, c2r, c3r = mr[cx][cy], mg[cx][cy], mg[cx][cy]

                if not shot and z < 1:
                    dtp = np.sqrt((x-posx)**2+(y-posy)**2+(z-posz)**2)
                    if dtp > 7:
                        modr = modr/np.log((dtp-6)/4+np.e)

                    if z < 1: # shadows
                        if sx != -1 and maph[int(sx)][int(sy)] > 1:
                            shot, c3 = 1, c3 * 0.9
                            dtol = np.sqrt((x-sx)**2+(y-sy)**2+(z-0.35)**2)
                            cos, sin, sinz = .01*(sx-x)/dtol, .01*(sy-y)/dtol, .01*(0.35-z)/dtol
                        else:
                            dtol = np.sqrt((x-lx)**2+(y-ly)**2+(z-lz)**2)
                            cos, sin, sinz = .01*(lx-x)/dtol, .01*(ly-y)/dtol, .01*(lz-z)/dtol
                        x += cos; y += sin; z += sinz
                        mapv = maph[int(x)][int(y)]
                        if z < mapv and mapv < 1 and not maps[int(x)][int(y)]:
                            modr = modr*0.39
                        while modr > 0.45:
                            if (mapv == 0) or not shot and ((z > mapv) or (z > 0.57 and mapv > 1)): ## LoDev DDA for optimization
                            
                                norm = np.sqrt(cos**2 + sin**2 + sinz**2)
                                rayDirX, rayDirY, rayDirZ = cos/norm + 1e-16, sin/norm + 1e-16, sinz/norm + 1e-16
                                
                                mapX, mapY = int(x), int(y)

                                deltaDistX, deltaDistY, deltaDistZ= abs(1/rayDirX), abs(1/rayDirY), abs(1/rayDirZ)

                                if (rayDirX < 0):
                                    stepX, sideDistX = -1, (x - mapX) * deltaDistX
                                else:
                                    stepX, sideDistX = 1, (mapX + 1.0 - x) * deltaDistX
                                    
                                if (rayDirY < 0):
                                    stepY, sideDistY = -1, (y - mapY) * deltaDistY
                                else:
                                    stepY, sideDistY = 1, (mapY + 1 - y) * deltaDistY

                                if (rayDirZ < 0):
                                    sideDistZ = z*deltaDistZ;
                                else:
                                    sideDistZ = (1-z)*deltaDistZ

                                while (1):
                                    if (sideDistX < sideDistY):
                                        sideDistX += deltaDistX; mapX += stepX
                                        dist = sideDistX; side = 0
                                        if mapX < 2 or mapX > size-2:
                                            break
                                    else:
                                        sideDistY += deltaDistY; mapY += stepY
                                        dist = sideDistY; side = 1
                                        if mapY < 2 or mapY > size-2:
                                            break
                                    if (maph[mapX][mapY] != 0):
                                        break
                                        
                                if (side):
                                    dist = dist - deltaDistY
                                else:
                                    dist = dist - deltaDistX
                                    
                                if (dist > sideDistZ):
                                    dist = sideDistZ

                                x = x + rayDirX*dist - cos/2
                                y = y + rayDirY*dist - sin/2
                                z = z + rayDirZ*dist - sinz/2
                            
                            ## end of LoDev DDA
                            x += cos; y += sin; z += sinz
                            mapv = maph[int(x)][int(y)]
                            if shot:
                                if mapv > 5 or (sinz > 0 and z > 0.35) or (sinz < 0 and z < 0.35):
                                    break
                            elif z >1:
                                break
                            if z < 0.57 and mapv > 1:
                                if  mapv == 3 or mapv == 9:
                                    if z> 0.45 and (x-enx)**2 + (y-eny)**2 + (z-0.5)**2 < 0.005 :
                                        modr = modr*0.67
                                    elif z < 0.45 and z > 0.3 and (x-enx)**2 + (y-eny)**2  < (z/10 - 0.02):
                                        modr = modr*0.67
                                    elif z < 0.3 and (x-enx)**2 + (y-eny)**2 + (z-0.15)**2 < 0.023 :
                                        modr = modr*0.67
                                elif mapv == 2 or mapv == 8:
                                    if z> 0.45 and (x-posx)**2 + (y-posy)**2 + (z-0.5)**2 < 0.005 :
                                        modr = modr*0.67
                                    elif z < 0.45 and z > 0.3 and (x-posx)**2 + (y-posy)**2  < (z/10 - 0.02):
                                        modr = modr*0.67
                                    elif z < 0.3 and (x-posx)**2 + (y-posy)**2 + (z-0.15)**2 < 0.023 :
                                        modr = modr*0.67
                                        
                            if mapv > 0 and z <= mapv and mapv < 2:      
                                if maps[int(x)][int(y)]: # check spheres
                                    if ((x-int(x)-0.5)**2 + (y-int(y)-0.5)**2 + (z-int(z)-0.5)**2 < 0.25):
                                        modr = modr*0.9
                                else:    
                                    modr = modr*0.9                    
                    
                pr[idx] = (3*modr*np.sqrt(c1*c1r) + (1-garbage)*pr[idx])/(4-garbage)
                pg[idx] = (3*modr*np.sqrt(c2*c2r) + (1-garbage)*pg[idx])/(4-garbage)
                pb[idx] = (3*modr*np.sqrt(c3*c3r) + (1-garbage)*pb[idx])/(4-garbage)
            idx += 1
    if checker: # fill gaps
        idx = 0
        for j in range(height): #vertical loop        
            for i in range(width): #horizontal vision loop
                if (i > 0 and i < width -1 and j > 0 and j < height -1 and ((inverse and i%2 != j%2) or (not(inverse) and i%2 == j%2))):
                    if abs(pr[idx-1] - pr[idx+1]) < 0.2 and abs(pg[idx-1] - pg[idx+1]) < 0.2 and abs(pb[idx-1] - pb[idx+1]) < 0.2 :
                        pr[idx] = (pr[idx-1] + pr[idx+1])/2
                        pg[idx] = (pg[idx-1] + pg[idx+1])/2
                        pb[idx] = (pb[idx-1] + pb[idx+1])/2
                    elif abs(pr[idx-width] - pr[idx+width]) < 0.2 and abs(pg[idx-width] - pg[idx+width]) < 0.2 and abs(pb[idx-width] - pb[idx+width]) < 0.2 :
                        pr[idx] = (pr[idx-width] + pr[idx+width])/2
                        pg[idx] = (pg[idx-width] + pg[idx+width])/2
                        pb[idx] = (pb[idx-width] + pb[idx+width])/2
                    else:
                        pr[idx] = ((1-garbage)*pr[idx] + pr[idx-1] + pr[idx-width] + pr[idx+width] + pr[idx+1])/(5-garbage)
                        pg[idx] = ((1-garbage)*pg[idx] + pg[idx-1] + pg[idx-width] + pg[idx+width] + pg[idx+1])/(5-garbage)
                        pb[idx] = ((1-garbage)*pb[idx] + pb[idx-1] + pb[idx-width] + pb[idx+width] + pb[idx+1])/(5-garbage)
                idx += 1
    return pr, pg, pb

def adjust_resol(width):
    height = int(0.6875*width)
    mod = width/64
    inc = 0.02/mod
    rr = np.random.uniform(0,1,width * height)
    gg = np.random.uniform(0,1,width * height)
    bb = np.random.uniform(0,1,width * height)
    return width, height, mod, inc, rr, gg, bb

@njit(cache=True)
def agents(enx, eny, maph, posx, posy, rot, et, shoot, sx, sy, sdir, shoot2, sx2, sy2, sdir2, mplayer, seenx, seeny, lock, size):
    
    if enx == 0 and np.random.uniform(0,1) > 0.995:
        x, y = np.random.normal(posx, 5), np.random.normal(posy, 5)
        dtp = (x-posx)**2 + (y-posy)**2
        if x > 0 and x < size-1 and y > 0 and y < size-1:
            if maph[int(x)][int(y)] == 0 and dtp > 25 and dtp < 64:
                enx, eny, seenx, seeny, lock = x, y, x, y, 0
    else:
        if not lock or  np.random.uniform(0,1) > 0.99:
            dtp = np.sqrt((enx-posx)**2 + (eny-posy)**2)
            cos, sin = (posx-enx)/dtp, (posy-eny)/dtp
            x, y = enx, eny
            for i in range(300):
                x += 0.04*cos; y += 0.04*sin
                if maph[int(x)][int(y)] != 0:
                    lock = 0
                    break
                if(int(x) == int(posx) and int(y) == int(posy)):
                    seenx, seeny = posx, posy
                    lock = 1
                    break

        if int(enx) == int(seenx) and int(eny) == int(seeny):
            if not lock:
                if shoot:
                    seenx, seeny = np.random.uniform(enx, posx), np.random.uniform(eny, posy)
                else:
                    seenx, seeny = np.random.normal(enx, 2), np.random.normal(eny, 2) 
            else:
                seenx, seeny = np.random.normal(posx, 2), np.random.normal(posy, 2)
            
        dtp = np.sqrt((enx-seenx)**2 + (eny-seeny)**2)    
        cos, sin = (seenx-enx)/dtp, (seeny-eny)/dtp    
        x, y = enx + et*(cos+np.random.normal(0,.5)), eny + et*(sin+np.random.normal(0,.5))

        if maph[int(x)][int(y)] == 0:
            enx, eny = x, y
        else:
            if np.random.uniform(0,1) > 0.5:
                x, y = enx - et*(sin+np.random.normal(0,.5)), eny + et*(cos+np.random.normal(0,.5))
            else:
                x, y = enx + et*(sin+np.random.normal(0,.5)), eny - et*(cos+np.random.normal(0,.5))
            if maph[int(x)][int(y)] == 0:
                enx, eny = x, y
            else:
                seenx, seeny = enx+np.random.normal(0,3), eny+np.random.normal(0,3)
                lock = 0
                
        mplayer[int(enx)][int(eny)] = 3

        
    mplayer[int(posx)][int(posy)] = 2
    if lock and not shoot2 and np.random.uniform(0,1) > 0.9:
        shoot2 = 1
        sdir2 = np.arctan((posy-eny)/(posx-enx)) + np.random.uniform(-.1,.1)
        if abs(enx+np.cos(sdir2)-posx) > abs(enx-posx):
            sdir2 = sdir2 - np.pi
        
    if shoot2:
        if sx2 == -1:
            sx2, sy2 = enx + .5*np.cos(sdir2), eny + .5*np.sin(sdir2)
        sx2, sy2 = sx2 + 5*et*np.cos(sdir2), sy2 + 5*et*np.sin(sdir2)
        if maph[int(sx2)][int(sy2)] != 0:
            shoot2, sx2, sy2 = 0, -1, -1
        else:    
            mplayer[int(sx2)][int(sy2)] += 6
    if shoot:
        if sx == -1:
            sdir = rot+np.random.uniform(-.1,.1)
            sx, sy = posx + .5*np.cos(sdir), posy + .5*np.sin(sdir)
        sx, sy = sx + 5*et*np.cos(sdir), sy + 5*et*np.sin(sdir)
        if enx != 0 and (sx - enx)**2 + (sy - eny)**2 < 0.02:
            shoot, sx, sy, enx, eny, seenx, seeny = 0, -1, -1, 0, 0, 0, 0
        if maph[int(sx)][int(sy)] != 0:
            shoot, sx, sy = 0, -1, -1
        else:    
            mplayer[int(sx)][int(sy)] += 12    
        
    
    mplayer = maph + mplayer
    return(enx, eny, mplayer, et, shoot, sx, sy, sdir, shoot2, sx2, sy2, sdir2, seenx, seeny, lock)

if __name__ == '__main__':
    main()
