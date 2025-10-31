# Quick Start Guide

## 🚀 Run the App

```bash
streamlit run smart_home_gui.py
```

Browser will open automatically at: `http://localhost:8501`

---

## 📱 What Each Field Means

| Field | What It Is | Range | Tips |
|-------|-----------|-------|------|
| 🏠 **Device Type** | Smart Speaker, Camera, Lights, Security System, or Thermostat | Select one | Choose the actual device type |
| 👤 **User ID** | Unique identifier for device owner | 1-100 | Can be any number |
| ⏰ **Usage Hours/Day** | How many hours per day device is used | 0-24 | Be realistic (12 = half day) |
| ⚡ **Energy Consumption** | How much power it uses | 0-10 | Higher = more power hungry |
| ✨ **User Preferences** | Simple or Advanced features | Low/High | High = more features enabled |
| 🔧 **Malfunction Incidents** | How many times it broke/failed | 0-10 | More problems = less efficient |
| 📅 **Device Age** | How old in months | 0-60 | Newer = more efficient |

---

## 📊 Navigation Menu

1. **Single Prediction** - Test one device
2. **Model Dashboard** - See all model performance
3. **Feature Importance** - What factors matter most
4. **Batch Prediction** - Test multiple devices at once

---

## ✅ Expected Results

```
🏠 Smart Speaker + 3 hrs/day + 1.5 energy = ✅ EFFICIENT (95%)
🏠 Camera + 18 hrs/day + 8.5 energy + 5 failures = ❌ INEFFICIENT (85%)
🏠 Thermostat + 24 hrs/day + 4.5 energy = ✅ EFFICIENT (70%)
```

---

## 💾 Export Your Results

- **Single Prediction**: Download as CSV with all parameters
- **Batch Prediction**: Download results with predictions for all devices

---

## 📈 Device Types

| Device | Typical Usage | Energy Level | Common Issues |
|--------|---------------|--------------|---------------|
| 🔊 Smart Speaker | 3-8 hrs/day | Low (1-3) | Connection, Voice recognition |
| 📷 Camera | 18-24 hrs/day | High (6-9) | Network, Recording |
| 💡 Lights | 8-12 hrs/day | Very Low (0.5-3) | Dimming, Scheduling |
| 🔒 Security System | 24 hrs/day | Low (1-4) | Sensor failure |
| 🌡️ Thermostat | 24 hrs/day | Medium (2-8) | Temp regulation |

---

## 🎯 Example: Test a Smart Speaker

1. Click **Single Prediction** tab
2. In sidebar:
   - **Device Type**: Smart Speaker
   - **User ID**: 5
   - **Usage Hours/Day**: 3
   - **Energy Consumption**: 1.5
   - **User Preferences**: Low
   - **Malfunction Incidents**: 0
   - **Device Age**: 6 months
3. Click **🔮 Make Prediction** button
4. See result and download if needed

---

## 📂 Batch Prediction Format

Create `devices.csv`:
```csv
DeviceType,UserID,UsageHoursPerDay,EnergyConsumption,UserPreferences,MalfunctionIncidents,DeviceAgeMonths
Smart Speaker,5,3,1.5,0,0,6
Camera,10,20,8,1,2,36
Lights,15,10,1,0,0,12
```

Then upload in **Batch Prediction** tab → Download results

---

## 🎓 How It Works

1. **You input** device parameters in human-readable format
2. **App converts** to normalized features (0-10)
3. **ML model predicts** if device is Efficient or Inefficient
4. **You get** prediction + confidence score

---

## ⚠️ Important Notes

- **Realistic values**: Use actual device parameters for accurate predictions
- **Confidence matters**: 80%+ is high confidence, below 60% is uncertain
- **Device tracking**: Update age and incident count as they change
- **Batch format**: CSV columns can be in any order, but device params should be realistic

---

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| App won't start | Run: `pip install -r requirements.txt` |
| Predictions seem wrong | Check if device params are realistic |
| Low confidence | May be a borderline case - unusual combination |
| CSV upload fails | Ensure all required columns present |

---

**Questions?** See `FEATURE_GUIDE.md` for detailed explanations!