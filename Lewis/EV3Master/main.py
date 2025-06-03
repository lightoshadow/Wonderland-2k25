#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile, Font
from pybricks.iodevices import UARTDevice
from pybricks.messaging import BluetoothMailboxServer, TextMailbox
import json


# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.
esp32 = UARTDevice(Port.S1, 115200)
server = BluetoothMailboxServer()
com1 = TextMailbox('com1', server)
com2 = TextMailbox('com2', server)
com3 = TextMailbox('com3', server)
dec = ""
ev3 = EV3Brick()

print('waiting for connection...')
ev3.screen.print('In attesa di connessioni')
server.wait_for_connection(3)
ev3.screen.print('CONNESSI')
print('connected!')


class msg_protocol:
    def __init__(self):
        self.tipo = 0 #0 = ESP32, 1 = EV3, 2 se EV3 Master
        self.ID = 0 #Se device_type è 0, questo è l'id dell'ESP32, se è 1, da 0 a 3 controlla i motori da A a D
        self.istruzione = "" #Messaggio da inviare a dispositivo. Valido solo se device_type = 0 o 1
        self.checkInvio = False

def write(messaggio):
    messaggio = messaggio + "!"
    esp32.write(messaggio.encode())

def write_prot(msg):
    write(json.dumps(msg.__dict__))
    print(json.dumps(msg.__dict__))

def readSER():
    msg = ""
    while True:
        char = esp32.read(1).decode()  # Leggi un carattere alla volta
        if char == "!":
            break  # Interrompi la lettura quando incontri "!"
        msg += char  # Aggiungi il carattere al messaggio
    
    esp32.clear()  # Pulisci il buffer di ricezione
    return msg

def conversioneDaJson(message):
    return json.loads(message)

def invioSER(ID, tipo, istruzione, checkInvio):
    test = msg_protocol()
    test.ID = ID
    test.tipo = tipo
    test.istruzione = istruzione
    test.checkInvio = checkInvio

    write_prot(test)

def invioBLT(ID, tipo, istruzione):
    checkInvio = False
    try:
        if istruzione != "":
            if ID == 1:
                com1.send(istruzione)
                com1.wait()
                risposta = com1.read()
                print(risposta)
                checkInvio = True
            elif ID == 2:
                com2.send(istruzione)
                com2.wait()
                risposta = com2.read()
                print(risposta)
                checkInvio = True
            elif ID == 3:
                com3.send(istruzione)
                com3.wait()
                risposta = com3.read()
                print(risposta)
                checkInvio = True
    except OSError as e:
        print("Errore: ", e)
        checkInvio = False
    return checkInvio

while True:
    print("entrato")
    letto = readSER()
    print(letto)
    messaggioPotabile = conversioneDaJson(letto)
    print(messaggioPotabile)
    ist = messaggioPotabile["istruzione"]
    ev3.screen.print(ist)

    print("\n",ist)

    if messaggioPotabile["tipo"] == 1:
        com1.send(ist)  
        invioSER(0,0,ist,True)
    elif messaggioPotabile["tipo"] == 2:
        com2.send(ist)
        invioSER(0,0,ist,True)
    elif messaggioPotabile["tipo"] == 3:
        com3.send(ist)
        invioSER(0,0,ist,True)
    else:
        print("Errore tipo non valido")
    
