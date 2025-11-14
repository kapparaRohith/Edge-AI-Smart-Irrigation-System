"""
USAGE INSTRUCTIONS:
===================

1. Basic Usage (Simulation Mode):
   python main_system.py

2. Hardware Mode (Raspberry Pi):
   - Edit config.json: set "mode": "production"
   - Ensure all sensors are connected
   - Run: python main_system.py

3. Background Execution:
   nohup python main_system.py > system.log 2>&1 &

4. Auto-start on Boot:
   sudo nano /etc/rc.local
   Add before 'exit 0':
   cd /home/pi/irrigation && python3 main_system.py &

5. Stop System:
   Press Ctrl+C (graceful shutdown)
   Or: pkill -f main_system.py

6. View Logs:
   tail -f logs/irrigation_*.log

7. Monitor MQTT:
   mosquitto_sub -v -t 'irrigation/#'

SYSTEM ARCHITECTURE:
====================

main_system.py (This file)
    ├── ml_models_implementation.py
    │   ├── LSTMPredictor
    │   ├── RandomForestDecision
    │   └── HybridIrrigationSystem
    │
    ├── hardware_iot_integration.py
    │   ├── SensorInterface
    │   ├── ActuatorController
    │   ├── MQTTCommunication
    │   └── EdgeDeviceManager
    │
    └── config.json
        └── System configuration

KEY FEATURES:
=============
✓ Real-time sensor data collection
✓ LSTM temporal prediction
✓ Random Forest decision making
✓ Autonomous irrigation control
✓ MQTT IoT communication
✓ Comprehensive logging
✓ System statistics tracking
✓ Graceful shutdown handling
✓ Edge computing optimization
✓ Water usage optimization

MONITORING ENDPOINTS:
=====================
- MQTT Topics:
  * irrigation/sensors/data - Raw sensor readings
  * irrigation/prediction - ML predictions
  * irrigation/status - System status
  * irrigation/control/# - Control commands

- Log Files:
  * logs/irrigation_YYYYMMDD.log - Daily logs

- System Stats:
  * Published every 10 cycles via MQTT
  * Includes uptime, predictions, water savings
"""
 
    """
    HARDWARE SETUP NOTES:
    
    Required Components:
    1. Raspberry Pi 4 (4GB recommended) or ESP32
    2. Sensors:
       - Capacitive Soil Moisture Sensor v1.2
       - DHT22 (Temperature & Humidity)
       - BH1750 (Light Intensity)
       - Tipping Bucket Rain Gauge
       - Anemometer (Wind Speed)
       - Analog pH Sensor
    
    3. Actuators:
       - 12V Water Pump
       - Solenoid Valve
       - Relay Module (2-channel)
    
    4. Additional:
       - MCP3008 ADC (for analog sensors)
       - 12V Power Supply
       - Breadboard & Jumper Wires
    
    Pin Connections (Raspberry Pi):
    - GPIO 17: Soil Moisture (via ADC)
    - GPIO 27: DHT22 Data
    - GPIO 22: Rain Sensor
    - GPIO 18: Pump Relay
    - GPIO 23: Valve Relay
    - SPI: MCP3008 ADC
    - I2C: BH1750 Light Sensor
    
    Software Requirements:
    pip install RPi.GPIO Adafruit-DHT paho-mqtt spidev smbus2
    
    MQTT Broker Setup:
    sudo apt-get install mosquitto mosquitto-clients
    sudo systemctl enable mosquitto
    sudo systemctl start mosquitto
    """
