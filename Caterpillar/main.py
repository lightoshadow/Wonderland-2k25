#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

import threading

ev3 = EV3Brick()

testa = Motor(Port.A)
corpo = Motor(Port.B)
fiore1 = Motor(Port.C)
fiore2 = Motor(Port.D)
dec = ""

# Connessione Bluetooth
client = BluetoothMailboxClient()
com3 = TextMailbox('com3', client)
SERVER = 'EV3Master'

# Connessione al server
ev3.screen.print('Connessione in corso...')
client.connect(SERVER)
ev3.screen.print('Connesso!')

class msg_protocol:
    def __init__(self):
        self.tipo = 0 #0 = ESP32, 1 = EV3, 2 se EV3 Master
        self.ID = 0 #Se device_type è 0, questo è l'id dell'ESP32, se è 1, da 0 a 3 controlla i motori da A a D
        self.istruzione = "" #Messaggio da inviare a dispositivo. Valido solo se device_type = 0 o 1
        self.checkInvio = False

def movimentoTesta():
    for a in range(1,15):
        testa.run(70)
        wait(1000)
        testa.stop()
        testa.run(-70)
        wait(1000)
        testa.stop()

def movimentoCorpo():
    corpo.run(367)
    # wait(30)
    # corpo.stop()

def movimentoFiore1():
    for a in range(1,5):
        fiore1.run(-1000)
        wait(3000)
        fiore1.stop()
        fiore1.run(1000)
        wait(3000)
        fiore1.stop()

def movimentoFiore2():
    for a in range(1,5):
        fiore2.run(-1000)
        wait(3000)
        fiore2.stop()
        fiore2.run(1000)
        wait(3000)
        fiore2.stop()

thread1 = threading.Thread(target=movimentoTesta)
thread2 = threading.Thread(target=movimentoCorpo)
thread3 = threading.Thread(target=movimentoFiore1)
thread4 = threading.Thread(target=movimentoFiore2)

def avvioProgramma():
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

while True:
    com3.wait()
    lett = com3.read()
    print(lett)
    ev3.screen.print(lett)

    ist = lett

    if ist == "scenaBruco":
        ev3.screen.print('avvioBruco')
        print(ist)
        avvioProgramma()
        wait(30000)