import bluetooth

# Scansione dispositivi
print("Scansione dispositivi BT...")
devices = bluetooth.discover_devices(duration=8, lookup_names=True)

for addr, name in devices:
    print(f"{addr} - {name}")

# Inserisci l’indirizzo del tuo ESP32
esp32_address = "CC:DB:27:9C:F7:30"  # Sostituisci con MAC ESP32

# Connetti alla porta SPP (canale 1 di solito)
port = 1
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

try:
    sock.connect((esp32_address, port))
    print("Connesso all'ESP32!")

    sock.send("Hello ESP32!\n")
    data = sock.recv(1024)
    print(f"Risposta: {data.decode('utf-8')}")

finally:
    sock.close()
