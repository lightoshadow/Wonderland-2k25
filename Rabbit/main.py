#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (
    Motor, ColorSensor, UltrasonicSensor
)
from pybricks.parameters import Port, Color
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.messaging import BluetoothMailboxClient, TextMailbox
import json

# Inizializza l'EV3 e i dispositivi
ev3 = EV3Brick()
left_motor = Motor(Port.D)
right_motor = Motor(Port.A)
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=104)

# Sensori
sensCentrale = UltrasonicSensor(Port.S2)
sensSX = UltrasonicSensor(Port.S1)
sensDX = UltrasonicSensor(Port.S3)
sensCol = ColorSensor(Port.S4)
testa = Motor(Port.B)
braccia = Motor(Port.C)

# Connessione Bluetooth
client = BluetoothMailboxClient()
com2 = TextMailbox('com2', client)
SERVER = 'EV3Master'

# Connessione al server
ev3.screen.print('Connessione in corso...')
client.connect(SERVER)
ev3.screen.print('Connesso!')

testa.run_target(100,0)
braccia.run_target(100,0)

print(testa.angle())
print(braccia.angle())

istruzione = "InviaTUTTO"
while True:
    letturaC = sensCentrale.distance()
    print("Centro:", letturaC)

    braccia.run_angle(200,60)
    braccia.run_angle(200,-60)

    if 20 < letturaC < 1000:
        robot.straight(-200)  # Avanza
    else:
        letturaS = sensSX.distance()
        print("Sinistra:", letturaS)
        if 20 < letturaS < 1000:
            testa.run_angle(100,-30)
            testa.run_angle(100,30)
            robot.turn(30)

        letturaD = sensDX.distance()
        print("Destra:", letturaD)
        if 20 < letturaD < 1000:
            testa.run_angle(100,30)
            testa.run_angle(100,-30)
            robot.turn(-30)

    # Se trova il colore blu o verde, si ferma e rompe il ciclo
    colore = sensCol.color()
    if colore == Color.BLUE or colore == Color.GREEN:
        robot.stop()
        ev3.screen.print('Colore trovato!')
        com2.send(istruzione)  # Invia il comando
        break

    wait(100)

# Se il comando è "avviaTUTTO", avvia il comportamento

    

# Una volta uscito dal ciclo, attende un nuovo comando
com2.wait()
istruzione = com2.read()
print("Secondo comando:", istruzione)
ev3.screen.print(istruzione)

# Se il comando è "giraConiglio", ruota su sé stesso
if istruzione == "giraConiglio":
    robot.turn(360)
