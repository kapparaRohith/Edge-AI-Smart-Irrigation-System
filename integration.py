"""
Hardware Integration and IoT Communication
For Raspberry Pi / ESP32 Edge Devices
"""

import time
import json
import threading
from datetime import datetime
import numpy as np

# MQTT for IoT communication
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("Warning: paho-mqtt not installed. Install with: pip install paho-mqtt")
    MQTT_AVAILABLE = False

# GPIO for Raspberry Pi (optional)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False


# ==================== SENSOR INTERFACE ====================
class SensorInterface:
    """
    Interface for various agricultural sensors
    """
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.sensors = {
            'soil_moisture': None,
            'temperature': None,
            'humidity': None,
            'rainfall': None,
            'light_intensity': None,
            'wind_speed': None,
            'soil_ph': None
        }
        
        if not simulation_mode and GPIO_AVAILABLE:
            self.setup_gpio()
        
    def setup_gpio(self):
        """Setup GPIO pins for sensors"""
        GPIO.setmode(GPIO.BCM)
        
        # Example pin assignments
        self.SOIL_MOISTURE_PIN = 17
        self.TEMP_HUMID_PIN = 27
        self.RAIN_PIN = 22
        
        # Setup pins
        GPIO.setup(self.SOIL_MOISTURE_PIN, GPIO.IN)
        GPIO.setup(self.RAIN_PIN, GPIO.IN)
        
    def read_soil_moisture_sensor(self):
        """
        Read soil moisture from capacitive sensor
        Typical sensors: Capacitive Soil Moisture Sensor v1.2
        """
        if self.simulation_mode:
            # Simulate realistic readings
            return 45 + np.random.normal(0, 5)
        
        # Real sensor reading (ADC conversion)
        # This would use SPI/I2C to read from ADC
        # Example for MCP3008 ADC:
        # adc_value = self.read_adc(channel=0)
        # moisture = (adc_value / 1023.0) * 100
        return 50.0
    
    def read_dht22_sensor(self):
        """
        Read temperature and humidity from DHT22 sensor
        """
        if self.simulation_mode:
            temp = 22 + np.random.normal(0, 2)
            humid = 65 + np.random.normal(0, 5)
            return temp, humid
        
        # Real DHT22 reading
        # import Adafruit_DHT
        # humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, self.TEMP_HUMID_PIN)
        return 22.0, 65.0
    
    def read_rain_sensor(self):
        """
        Read rainfall from tipping bucket rain gauge
        """
        if self.simulation_mode:
            return np.random.exponential(1) if np.random.random() > 0.8 else 0
        
        # Real rain sensor (counts tips)
        # Each tip = 0.2794mm of rain
        return 0.0
    
    def read_light_sensor(self):
        """
        Read light intensity from BH1750 sensor
        """
        if self.simulation_mode:
            hour = datetime.now().hour
            # Simulate day/night cycle
            if 6 <= hour <= 18:
                return 60 + 30 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 5)
            return np.random.normal(10, 3)
        
        # Real BH1750 sensor
        return 75.0
    
    def read_wind_sensor(self):
        """
        Read wind speed from anemometer
        """
        if self.simulation_mode:
            return abs(np.random.normal(5, 2))
        return 5.0
    
    def read_ph_sensor(self):
        """
        Read soil pH from analog pH sensor
        """
        if self.simulation_mode:
            return np.random.normal(6.5, 0.3)
        return 6.5
    
    def read_all_sensors(self):
        """
        Read all sensors and return data dict
        """
        temp, humid = self.read_dht22_sensor()
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'soil_moisture': float(self.read_soil_moisture_sensor()),
            'temperature': float(temp),
            'humidity': float(humid),
            'rainfall': float(self.read_rain_sensor()),
            'light_intensity': float(self.read_light_sensor()),
            'wind_speed': float(self.read_wind_sensor()),
            'soil_ph': float(self.read_ph_sensor())
        }
        
        return data
    
    def cleanup(self):
        """Cleanup GPIO"""
        if GPIO_AVAILABLE and not self.simulation_mode:
            GPIO.cleanup()


# ==================== ACTUATOR CONTROLLER ====================
class ActuatorController:
    """
    Control irrigation pump and valves
    """
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.pump_active = False
        
        if not simulation_mode and GPIO_AVAILABLE:
            self.setup_actuators()
    
    def setup_actuators(self):
        """Setup GPIO pins for actuators"""
        self.PUMP_PIN = 18
        self.VALVE_PIN = 23
        
        GPIO.setup(self.PUMP_PIN, GPIO.OUT)
        GPIO.setup(self.VALVE_PIN, GPIO.OUT)
        
        # Initial state: OFF
        GPIO.output(self.PUMP_PIN, GPIO.LOW)
        GPIO.output(self.VALVE_PIN, GPIO.LOW)
    
    def activate_pump(self, duration_seconds=300):
        """
        Activate irrigation pump for specified duration
        
        Args:
            duration_seconds: How long to run pump (default 5 minutes)
        """
        if self.simulation_mode:
            print(f"[SIMULATION] Pump activated for {duration_seconds}s")
            self.pump_active = True
            return
        
        if GPIO_AVAILABLE:
            print(f"Activating pump for {duration_seconds} seconds...")
            GPIO.output(self.PUMP_PIN, GPIO.HIGH)
            GPIO.output(self.VALVE_PIN, GPIO.HIGH)
            self.pump_active = True
            
            # Use threading to avoid blocking
            def turn_off():
                time.sleep(duration_seconds)
                self.deactivate_pump()
            
            threading.Thread(target=turn_off, daemon=True).start()
    
    def deactivate_pump(self):
        """Deactivate irrigation pump"""
        if self.simulation_mode:
            print("[SIMULATION] Pump deactivated")
            self.pump_active = False
            return
        
        if GPIO_AVAILABLE:
            print("Deactivating pump...")
            GPIO.output(self.PUMP_PIN, GPIO.LOW)
            GPIO.output(self.VALVE_PIN, GPIO.LOW)
            self.pump_active = False
    
    def get_status(self):
        """Get current pump status"""
        return {
            'pump_active': self.pump_active,
            'timestamp': datetime.now().isoformat()
        }


# ==================== MQTT COMMUNICATION ====================
class MQTTCommunication:
    """
    MQTT client for IoT communication
    """
    def __init__(self, broker='localhost', port=1883, client_id='irrigation_edge'):
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.client = None
        self.connected = False
        
        if MQTT_AVAILABLE:
            self.setup_client()
    
    def setup_client(self):
        """Setup MQTT client"""
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for connection"""
        if rc == 0:
            print(f"Connected to MQTT broker at {self.broker}:{self.port}")
            self.connected = True
            
            # Subscribe to control topics
            self.client.subscribe("irrigation/control/#")
            self.client.subscribe("irrigation/config/#")
        else:
            print(f"Connection failed with code {rc}")
            self.connected = False
    
    def on_message(self, client, userdata, msg):
        """Callback for received messages"""
        print(f"Received: {msg.topic} - {msg.payload.decode()}")
        
        # Handle control messages
        if msg.topic == "irrigation/control/pump":
            command = msg.payload.decode()
            if command == "ON":
                print("Remote pump activation command received")
            elif command == "OFF":
                print("Remote pump deactivation command received")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        print(f"Disconnected from MQTT broker (code: {rc})")
        self.connected = False
    
    def connect(self):
        """Connect to MQTT broker"""
        if not MQTT_AVAILABLE:
            print("MQTT not available")
            return False
        
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False
    
    def publish_sensor_data(self, sensor_data):
        """Publish sensor data to MQTT"""
        if not self.connected:
            return False
        
        topic = "irrigation/sensors/data"
        payload = json.dumps(sensor_data)
        
        result = self.client.publish(topic, payload, qos=1)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    
    def publish_prediction(self, prediction_data):
        """Publish ML prediction to MQTT"""
        if not self.connected:
            return False
        
        topic = "irrigation/prediction"
        payload = json.dumps(prediction_data)
        
        result = self.client.publish(topic, payload, qos=1)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    
    def publish_status(self, status_data):
        """Publish system status to MQTT"""
        if not self.connected:
            return False
        
        topic = "irrigation/status"
        payload = json.dumps(status_data)
        
        result = self.client.publish(topic, payload, qos=1)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# ==================== EDGE DEVICE MANAGER ====================
class EdgeDeviceManager:
    """
    Main manager for edge device operations
    """
    def __init__(self, simulation_mode=True, mqtt_broker='localhost'):
        self.simulation_mode = simulation_mode
        self.sensors = SensorInterface(simulation_mode=simulation_mode)
        self.actuators = ActuatorController(simulation_mode=simulation_mode)
        self.mqtt = MQTTCommunication(broker=mqtt_broker)
        
        self.running = False
        self.sensor_interval = 60  # Read sensors every 60 seconds
        self.data_history = []
        
    def start(self):
        """Start edge device operations"""
        print("\n" + "=" * 60)
        print("STARTING EDGE DEVICE")
        print("=" * 60)
        print(f"Mode: {'SIMULATION' if self.simulation_mode else 'HARDWARE'}")
        
        # Connect to MQTT
        if MQTT_AVAILABLE:
            self.mqtt.connect()
        
        self.running = True
        
        print("Edge device started successfully!")
        print("Press Ctrl+C to stop\n")
        
        # Start main loop
        try:
            self.main_loop()
        except KeyboardInterrupt:
            print("\n\nShutdown signal received...")
            self.stop()
    
    def main_loop(self):
        """Main operational loop"""
        while self.running:
            # Read sensors
            sensor_data = self.sensors.read_all_sensors()
            
            print(f"\n[{sensor_data['timestamp']}]")
            print(f"Soil Moisture: {sensor_data['soil_moisture']:.1f}%")
            print(f"Temperature: {sensor_data['temperature']:.1f}°C")
            print(f"Humidity: {sensor_data['humidity']:.1f}%")
            print(f"Rainfall: {sensor_data['rainfall']:.1f}mm")
            
            # Store history
            self.data_history.append(sensor_data)
            if len(self.data_history) > 100:
                self.data_history.pop(0)
            
            # Publish sensor data
            if MQTT_AVAILABLE and self.mqtt.connected:
                self.mqtt.publish_sensor_data(sensor_data)
            
            # Get actuator status
            status = self.actuators.get_status()
            
            # Publish status
            if MQTT_AVAILABLE and self.mqtt.connected:
                self.mqtt.publish_status(status)
            
            # Wait for next reading
            time.sleep(self.sensor_interval)
    
    def make_irrigation_decision(self, ml_system, sensor_data):
        """
        Use ML system to make irrigation decision
        
        Args:
            ml_system: HybridIrrigationSystem instance
            sensor_data: Current sensor readings
        """
        try:
            # Need recent 24-hour data for LSTM
            if len(self.data_history) >= 24:
                import pandas as pd
                recent_df = pd.DataFrame(self.data_history[-24:])
                
                # Get prediction
                result = ml_system.predict(recent_df, sensor_data)
                
                print(f"\n--- ML PREDICTION ---")
                print(f"Predicted Moisture: {result['predicted_moisture']:.1f}%")
                print(f"Irrigation Needed: {result['irrigation_decision']}")
                print(f"Confidence: {result['confidence']*100:.1f}%")
                
                # Publish prediction
                if MQTT_AVAILABLE and self.mqtt.connected:
                    self.mqtt.publish_prediction(result)
                
                # Execute irrigation if needed
                if result['irrigation_decision'] and not self.actuators.pump_active:
                    print("\n✓ Activating irrigation based on ML decision")
                    self.actuators.activate_pump(duration_seconds=300)
                
                return result
            else:
                print("Collecting initial data... (need 24 readings)")
                return None
                
        except Exception as e:
            print(f"Error in ML decision: {e}")
            return None
    
    def stop(self):
        """Stop edge device operations"""
        print("Stopping edge device...")
        self.running = False
        
        # Cleanup
        self.actuators.deactivate_pump()
        self.sensors.cleanup()
        
        if MQTT_AVAILABLE:
            self.mqtt.disconnect()
        
        print("Edge device stopped successfully")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("Edge Device - Hardware & IoT Integration")
    print("For Smart Irrigation System\n")
    
    # Initialize edge device in simulation mode
    edge_device = EdgeDeviceManager(
        simulation_mode=True,  # Set to False for real hardware
        mqtt_broker='localhost'  # Change to your MQTT broker address
    )
    
    # Start device operations
    edge_device.start()
   
