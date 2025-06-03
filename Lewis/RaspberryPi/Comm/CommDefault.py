import serial, json

ser = serial.Serial(
    port = "/dev/ttyUSB0",
    baudrate = 115200,
    timeout = 1
)
class msg_protocol:
    def __init__(self):
        self.tipo = 0 #0 = ESP32, 1 = EV3, 2 se EV3 Master, 4 RASPI
        self.ID = 0 #Se device_type è 0, questo è l'id dell'ESP32, se è 1, da 0 a 3 controlla i motori da A a D
        self.istruzione = "" #Messaggio da inviare a dispositivo. Valido solo se device_type = 0 o 1
        self.checkInvio = False
    def __str__(self):
        return "{\"tipo\" : " + str(self.tipo) + ", \"ID\" : " + str(self.ID) + ", \"istruzione\" : " + str(self.istruzione) + " , \"CheckInvio\" : " + str(self.checkInvio) + "}"

class MSG_COMM_ERROR(Exception):
    def __init__(self, message, recvd):
        self.message = message
        self.recvd = recvd
    def __str__(self):
        return self.message

def read(ser):
    recv = False
    while True:
        b = ser.read()
        if b == b"{":
            recv = True
            s = ""
        if recv: s += b.decode()
        if b == b"!":
            recv = False
            return s[0:-1]

def write(ser, msg):
    ser.write(f"{msg}!".encode())

def write_prot(ser, msg):
    write(ser, json.dumps(msg.__dict__))

def invioSER(ser, ID, tipo, istruzione):
    test = msg_protocol()
    test.ID = ID
    test.tipo = tipo
    test.istruzione = istruzione
    test.checkInvio = False
    print(test)        
    write_prot(ser, test)
    msg = read_prot(ser)
    print(msg)
    test.checkInvio = True
    if test == msg:
        return True
    else:
        raise MSG_COMM_ERROR("Errore durante la comunicazione!", msg)


def read_prot(ser):
    msg = msg_protocol()
    msg_s = read(ser)
    msg_j = json.loads(msg_s)
    msg.ID = msg_j["ID"]
    msg.tipo = msg_j["tipo"]
    msg.istruzione = msg_j["istruzione"]
    msg.checkInvio = True
    write_prot(ser, msg)
    return msg