# Wonderland-2k25

Repository for the Wonderland team's entry in the RoboCup OnStage 2025 Competition. This project brings characters from "Alice in Wonderland" to life through a coordinated robotic performance.

## Project Overview

The "Wonderland-2k25" performance features several autonomous robots, each representing a character from Alice in Wonderland. The entire show is orchestrated by a central system named **Lewis**, which uses image recognition (via Raspberry Pi) to identify scenes and then dispatches commands to the various character robots through a network of ESP32 microcontrollers and EV3 bricks.

The general control flow is:
1.  **Lewis (Raspberry Pi)**: Detects scenes using a camera and an image recognition model. It initiates the sequence upon detecting a "START" QR code.
2.  **Lewis (ESPMaster - ESP32)**: Receives scene information from the Raspberry Pi via UART.
3.  **ESPMaster**: Acts as the primary command hub, coordinating actions by:
    *   Sending commands directly via Bluetooth to specific ESP32 slaves (e.g., Queen of Hearts' ESP32).
    *   Sending commands via UART to an **ESPIntermezzo (ESP32)**, which then relays them via Bluetooth to other ESP32 slaves (e.g., Cheeshire Cat's ESP32). This helps manage Bluetooth connections.
    *   Sending commands to the **Lewis (EV3Master - EV3 Brick)** via UART.
4.  **EV3Master**: Relays commands received from the ESPMaster to individual EV3-based character robots (Caterpillar, Rabbit, Queen's EV3 components) via Bluetooth mailboxes.
5.  **Character Robots**: Each robot (Caterpillar, Cheeshire Cat, Queen of Hearts, Rabbit) executes its pre-programmed routines and movements based on the commands received from its respective controller (EV3 client or ESP32 slave).

## Key Technologies

*   **Robotics Platforms:**
    *   LEGO Mindstorms EV3
    *   ESP32 Microcontrollers
    *   Raspberry Pi
*   **Programming Languages:**
    *   MicroPython (for EV3 bricks)
    *   Arduino C++ (for ESP32s, using PlatformIO)
    *   Python 3 (for Raspberry Pi and development tools)
*   **Communication Protocols:**
    *   Bluetooth (EV3-to-EV3, ESP32-to-ESP32)
    *   UART Serial (Raspberry Pi-to-ESP32, ESP32-to-EV3, ESP32-to-ESP32)
    *   JSON for message formatting between devices.
*   **Machine Learning:**
    *   PyTorch with EfficientNet (for image recognition on Raspberry Pi).
*   **Development Environment:**
    *   PlatformIO (for ESP32 development)
    *   Visual Studio Code

## Directory Structure

The project is organized into folders, each representing a major component or character robot:

*   `README.md`: This file, providing a project overview.
*   `Caterpillar/`: Contains the MicroPython code for the EV3-based Caterpillar robot.
*   `CheeshireCat/`: Contains code for the Cheeshire Cat, which utilizes both an ESP32 (for complex movements/effects) and an EV3 brick (for communication relay and potentially other motors).
*   `devTools/`: Includes development scripts, a Jupyter/Colab notebook for training the image recognition model (`TrainingWonderland.ipynb`), and potentially other utility code. The `progESPMasterAggiornatoDAFINIRE.ino` is likely an older or developmental version of an ESP32 master controller.
*   `Lewis/`: The core of the orchestration system.
    *   `Lewis/RaspberryPi/`: Python code for image recognition and high-level scene detection.
    *   `Lewis/ESPMaster/`: Arduino C++ code for the main ESP32 controller that interfaces with the Raspberry Pi, other ESP32s, and the EV3Master.
    *   `Lewis/ESPintermezzo/`: Arduino C++ code for a secondary ESP32 acting as a communication bridge.
    *   `Lewis/EV3Master/`: MicroPython code for the EV3 brick that acts as a Bluetooth master for other EV3 robots.
*   `QueenOfHearts/`: Code for the Queen of Hearts robot, which uses an ESP32 (for arm movements and LEDs) and an EV3 brick (for skirt movement).
*   `Rabbit/`: Contains the MicroPython code for the EV3-based Rabbit robot.

## Further Information

Each main component/character folder (`Caterpillar/`, `CheeshireCat/`, `Lewis/`, `QueenOfHearts/`, `Rabbit/`, `devTools/`) contains its own `README.md` file with more specific details about its functionality, sub-components, code structure, and setup instructions. Please refer to those for in-depth information.
