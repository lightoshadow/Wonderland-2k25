## Code to be run on the raspberry pi

main script is `detection.py` to run it needs:  
- `labels.txt`  
- `WonderlandEfficentNet.pth`  
- `./audios`  
- `./Comm`
  
The main script loads an image recognition model in memory (EfficentNet) then it tries to detect a QR code, after it has detected it he starts to run inference in the model. Based on the detection it sends messages to the ESP32 so that it can communicate with all the other robots.

`/Comm` is a python module used to setup UART communication between the RaspberryPi and the ESP Master
