# All of the code used for the Queen Of Hearts

This folder contains the software for the Queen of Hearts robot.

-   `/ESP32`: Code running on the ESP32 microcontroller, responsible for controlling the Queen of Hearts' specific actions like arm movements and LED eye colors. It communicates via Bluetooth with the ESPMaster.
-   `/EV3`: MicroPython code running on the EV3 brick, dedicated to controlling parts of the Queen of Hearts robot, such as the "gonna" (skirt) motor. It communicates via Bluetooth with the EV3Master.
