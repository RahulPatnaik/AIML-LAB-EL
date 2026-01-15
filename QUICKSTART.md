# ⚡ QUICKSTART CHECKLIST

## 🎯 Complete in 5 Minutes

### ✅ Step 1: Install Dependencies (2 minutes)

```bash
cd "AIML LAB EL/backend"
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed fastapi-0.100.0 uvicorn-0.23.0 shap-0.42.0 ...
```

---

### ✅ Step 2: Start Backend (30 seconds)

```bash
python main.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
```

✅ Backend is running when you see "Uvicorn running"

**Keep this terminal open!**

---

### ✅ Step 3: Start Frontend (30 seconds)

**Open a NEW terminal:**

```bash
cd "AIML LAB EL/frontend"
python3 -m http.server 3000
```

**Expected Output:**
```
Serving HTTP on 0.0.0.0 port 3000 (http://0.0.0.0:3000/) ...
```

✅ Frontend is running

---

### ✅ Step 4: Open Dashboard (10 seconds)

Open your browser and go to:

**http://localhost:3000**

You should see:
- 🧠 Purple gradient header
- Dropdown with 17 disorders
- 5 colorful buttons

---

### ✅ Step 5: Test the App (1 minute)

1. **Make a Prediction:**
   - Click "🎯 Predict" button
   - Wait 2 seconds
   - See prediction card appear with confidence levels

2. **Try SHAP Explanation:**
   - Click "📊 SHAP Explanation"
   - Wait 5 seconds
   - See bar chart with top features

3. **Explore Other Features:**
   - Try "🔍 LIME Explanation"
   - Try "🗺️ Feature Space" (PCA visualization)
   - Try "🌳 Decision Path" (tree analysis)

---

## ✅ Verification Checklist

Mark each when working:

- [ ] Backend starts without errors
- [ ] Frontend serves at port 3000
- [ ] Dashboard loads in browser
- [ ] Dropdown shows 17 disorders
- [ ] Predict button returns results
- [ ] SHAP shows bar chart
- [ ] LIME shows explanations
- [ ] Feature space shows PCA plot
- [ ] Decision path shows tree info

---

## 🐛 Quick Troubleshooting

### Problem: "Module not found: numpy"
**Solution:**
```bash
pip install -r backend/requirements.txt
```

### Problem: "Port 8000 already in use"
**Solution:**
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Then restart backend
```

### Problem: "Models not found"
**Solution:**
Check that `all_models/` directory exists with .pkl files:
```bash
ls all_models/*.pkl | wc -l
# Should show ~85 files
```

### Problem: Frontend shows CORS error
**Solution:**
- Ensure backend is on port 8000
- Ensure frontend is on port 3000
- Refresh browser (Ctrl+Shift+R)

### Problem: SHAP/LIME not working
**Solution:**
```bash
pip install shap==0.42.0 lime==0.2.0.1 --force-reinstall
```

---

## 🎉 Success!

If you can see predictions and explanations, **you're done!**

You now have a fully functional AI explainability dashboard.

---

## 📚 Next Steps

1. **Read Documentation:**
   - `SETUP_GUIDE.md` - Detailed setup
   - `README_EXPLAINABILITY.md` - All features
   - `ARCHITECTURE.md` - How it works

2. **Customize:**
   - Try different disorders
   - Modify sample patient in `backend/sample_data.csv`
   - Change UI colors in `frontend/style.css`

3. **Extend:**
   - Add more visualization options
   - Create PDF export
   - Add authentication
   - Connect to real EEG device

---

## 🆘 Still Stuck?

1. Check Python version: `python --version` (need 3.8+)
2. Check pip version: `pip --version`
3. View backend logs in terminal 1
4. Check browser console (F12) for JavaScript errors
5. Read full documentation in README_EXPLAINABILITY.md

---

**Time to Complete:** ~5 minutes
**Difficulty:** ⭐⭐☆☆☆ (Easy)
**Requirements:** Python 3.8+, pip, browser

**Ready to explore AI explainability? Let's go! 🚀**
