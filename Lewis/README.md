# All of the code used for Lewis

Lewis is the central orchestrator of the Wonderland performance. This directory contains the code for the various components that make up Lewis's control system.

-   **/RaspberryPi**
    This folder houses the Python scripts running on the Raspberry Pi, which is responsible for visual processing and high-level decision making.

-   **/ESPMaster**
    This folder contains the Arduino C++ code for the primary ESP32 microcontroller, acting as the main communication hub and command dispatcher for Lewis.

-   **/ESPintermezzo**
    This folder contains the Arduino C++ code for a secondary ESP32 microcontroller that acts as a communication bridge or relay.

-   **/EV3Master**
    This folder contains the MicroPython code for the LEGO EV3 brick that serves as a master controller for other EV3-based robots in the performance.
