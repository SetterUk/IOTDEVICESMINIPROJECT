# Smart Home Device Efficiency Predictor - Feature Guide

## Overview
This ML predictor determines whether a smart home device is **Efficient** or **Inefficient** based on real device parameters.

---

## 🏠 Device Types

The app supports 5 different smart home device types:

### 1. **Smart Speaker** 🔊
- Voice-controlled devices (Alexa, Google Home)
- Typical usage: Playing music, setting reminders, voice commands
- Energy: Low to medium (1-8 units)
- Common issues: Connection problems, voice recognition failures

### 2. **Camera** 📷
- Security/surveillance cameras
- Typical usage: 24/7 monitoring or scheduled recording
- Energy: Medium to high (2-9 units)
- Common issues: Network connectivity, recording failures

### 3. **Lights** 💡
- Smart LED bulbs and lighting systems
- Typical usage: 8-12 hours per day
- Energy: Very low (0.5-3 units)
- Common issues: Dimming failures, scheduling issues

### 4. **Security System** 🔒
- Door/window sensors, alarm systems
- Typical usage: 24/7 active
- Energy: Low (1-4 units)
- Common issues: Sensor malfunction, false alarms

### 5. **Thermostat** 🌡️
- Temperature control systems
- Typical usage: Continuous (24 hours)
- Energy: Medium (2-8 units)
- Common issues: Heating/cooling inefficiency, schedule bugs

---

## 📊 Device Parameters

### **👤 User ID** (1-100)
- Identifier for the device owner
- Different users may have different usage patterns
- Default: 5

### **⏰ Usage Hours/Day** (0-24 hours)
- How many hours per day the device is actively used
- **Low**: 0-8 hours (occasional use)
- **Medium**: 8-16 hours (regular use)
- **High**: 16-24 hours (continuous/heavy use)
- Default: 12 hours

**Impact on Efficiency:**
- Too low: Device may not be properly utilized
- Too high: May indicate power waste or constant use

### **⚡ Energy Consumption** (0-10 units)
- Measured power usage of the device
- **Low** (0-3): Energy-efficient devices
- **Medium** (3-6): Average consumption
- **High** (6-10): Power-hungry devices
- Default: 5 units

**Impact on Efficiency:**
- Lower consumption = More likely efficient
- Higher consumption = Efficiency depends on usage

### **✨ User Preferences** (Low / High)
- User's preference setting or configuration level
- **Low (0)**: Basic functionality, minimal customization
- **High (1)**: Advanced features, full customization enabled
- Default: Low

**Impact on Efficiency:**
- Advanced features may increase power consumption
- Simpler settings tend to be more efficient

### **🔧 Malfunction Incidents** (0-10)
- Number of times the device has malfunctioned/failed
- **0**: Never had issues
- **1-3**: Minor occasional issues
- **4-6**: Regular problems
- **7+**: Frequent failures
- Default: 2 incidents

**Impact on Efficiency:**
- More malfunctions = Likely inefficient
- Devices with issues often waste energy

### **📅 Device Age** (0-60 months)
- How old the device is in months
- **New** (0-6 months): Should be efficient
- **Recent** (6-24 months): Good performance expected
- **Aging** (24-48 months): Degradation starting
- **Old** (48+ months): Efficiency declining
- Default: 24 months

**Impact on Efficiency:**
- Newer devices tend to be more efficient
- Older devices accumulate wear and reduce efficiency

---

## 🔮 Prediction Output

### **Efficiency Status**
- **✅ Efficient**: Device is working well with good energy management
- **❌ Inefficient**: Device is consuming power wastefully or malfunctioning

### **Confidence Score**
- **80-100%**: High confidence prediction (very reliable)
- **60-80%**: Medium confidence (reasonably reliable)
- **Below 60%**: Low confidence (may need verification)

### **Prediction Code**
- **1**: Efficient
- **0**: Inefficient

---

## 📈 Example Scenarios

### Scenario 1: New Efficient Smart Speaker
```
Device Type: Smart Speaker
User ID: 10
Usage Hours/Day: 3 hours
Energy Consumption: 1.5 units
User Preferences: Low
Malfunction Incidents: 0
Device Age: 2 months
Expected Result: ✅ Efficient (95%+)
```

### Scenario 2: Old Camera with Issues
```
Device Type: Camera
User ID: 45
Usage Hours/Day: 18 hours
Energy Consumption: 8.5 units
User Preferences: High
Malfunction Incidents: 5
Device Age: 54 months
Expected Result: ❌ Inefficient (85%+)
```

### Scenario 3: Average Thermostat
```
Device Type: Thermostat
User ID: 30
Usage Hours/Day: 24 hours
Energy Consumption: 4.5 units
User Preferences: Medium
Malfunction Incidents: 1
Device Age: 18 months
Expected Result: ✅ Efficient (70%)
```

---

## 💾 Batch Prediction Format

For **Batch Prediction**, upload a CSV with these columns (in any order):

```csv
DeviceType,UserID,UsageHoursPerDay,EnergyConsumption,UserPreferences,MalfunctionIncidents,DeviceAgeMonths
Smart Speaker,5,12,2,0,0,24
Camera,10,18,8,1,3,36
Lights,15,8,1,0,0,6
```

The app will:
1. Validate your data
2. Normalize the features
3. Make predictions for each device
4. Return results with confidence scores
5. Allow you to download the results as CSV

---

## 🎯 Tips for Better Predictions

1. **Use accurate data** - The more precise your input values, the better the prediction
2. **Keep tracking** - Monitor device age and malfunction incidents regularly
3. **Consider efficiency** - Devices with many incidents should be repaired or replaced
4. **Balance usage** - Very high usage hours may indicate inefficient operation
5. **Check energy levels** - Unusually high energy consumption warrants investigation

---

## 📊 Model Information

- **Models Trained**: 9 different ML algorithms
- **Primary Model**: Decision Tree (used for predictions)
- **Accuracy**: ~85-90% on test data
- **Training Data**: 5,403 smart home device records
- **Device Coverage**: All major smart home device types

---

## ❓ Troubleshooting

### Prediction seems wrong?
- Check if device parameters are realistic
- Verify device type is correct
- Ensure age and malfunction incidents are accurate

### Confidence score too low?
- This may be a borderline case
- Consider collecting more data about the device
- Check if there are unusual parameter combinations

### Need more help?
- Review the example scenarios above
- Check the Model Dashboard for comparison metrics
- Look at Feature Importance to understand key factors

---

**Ready to use?** Run: `streamlit run smart_home_gui.py`