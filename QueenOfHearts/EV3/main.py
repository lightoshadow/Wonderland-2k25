#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile, Font
from pybricks.iodevices import UARTDevice
import json
from pybricks.messaging import BluetoothMailboxClient, TextMailbox

# This is the name of the remote EV3 or PC we are connecting to.
SERVER = 'EV3Master'

client = BluetoothMailboxClient()


ev3 = EV3Brick()
gonna = Motor(Port.A)
dec = ""

ev3.screen.print('Connessione in corso')
client.connect(SERVER)
ev3.screen.print('CONNESSO')


com1 = TextMailbox('com1', client)

class msg_protocol:
    def __init__(self):
        self.tipo = 0 #0 = ESP32, 1 = EV3, 2 se EV3 Master
        self.ID = 0 #Se device_type è 0, questo è l'id dell'ESP32, se è 1, da 0 a 3 controlla i motori da A a D
        self.istruzione = "" #Messaggio da inviare a dispositivo. Valido solo se device_type = 0 o 1
        self.checkInvio = False

def conversioneDaJson(message):
    return json.loads(message)

def conversioneAJson(message):
    return json.dumps(message.__dict__)

def risposta(ID, tipo, istruzione):
    msg = msg_protocol()
    msg.ID = ID
    msg.tipo = tipo
    msg.istruzione = istruzione
    msg.checkInvio = False

    return conversioneAJson(msg)

statusGonna = False

while True:
    com1.wait()
    lett = com1.read()
    print(lett)
    ev3.screen.print(lett)

    ist = lett

    if ist == "avvioGonna":
        ev3.screen.print('avvioGonna')
        statusGonna = True
        gonna.run(200)
        print(ist)

    if ist == "arrestoGonna":
        ev3.screen.print('arrestoGonna')
        statusGonna = False
        gonna.stop()
        print(ist)


    #com1.send(risposta())
