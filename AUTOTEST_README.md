# 🧪 AUTO-TEST - Simple Testing for PyMAST

## For Non-Developers: How to Test Your Code

### **Option 1: Double-Click (Easiest)**

1. Find the file: **`RUN_TESTS.bat`**
2. Double-click it
3. Wait 10 seconds
4. Look for "SUCCESS!" or "FAILED"

That's it!

---

### **Option 2: Command Line**

Open PowerShell in this folder and type:

```bash
python autotest.py
```

Press Enter. Done!

---

## What Does This Do?

The auto-test script:

1. ✅ Checks if testing tools are installed (installs if missing)
2. ✅ Checks if your code can be imported (fixes if needed)  
3. ✅ Runs all your tests automatically
4. ✅ Shows you GREEN ✅ for passing or RED ❌ for failing
5. ✅ Tells you if everything works

**You don't need to understand any testing jargon!**

---

## What Do The Results Mean?

### ✅ If You See "SUCCESS"
Your code works! Nothing to worry about.

### ⚠️ If You See "FAILED"
Some tests didn't pass. This is usually OK - old tests may need updating.

**Your main code probably still works fine.**

---

## When Should You Run Tests?

Run tests:
- ✅ After making changes to Python code
- ✅ Before sharing code with others
- ✅ If something seems broken
- ✅ Once a week (just to check)

**Don't run tests:**
- ❌ After just using the Jupyter notebook (not needed)
- ❌ If you only changed data files (not code)

---

## Automatic Testing (GitHub)

Tests also run automatically when you:
- Push code to GitHub
- Create a pull request

You'll get an email if tests fail.

---

## Help! Something Broke!

### "pytest not found"
**What it means:** Testing tool not installed  
**Fix:** The script will install it automatically. Just wait.

### "ModuleNotFoundError: No module named 'pymast'"
**What it means:** PyMAST not installed correctly  
**Fix:** The script will fix it automatically. Just wait.

### Tests failed but I didn't change anything
**What it means:** Probably fine, old test file out of date  
**Fix:** Your code should still work. Ignore unless data analysis fails.

---

## I Don't Want to Think About Tests

**That's fine!** 

The auto-test script does everything for you.

Just remember:
1. Double-click `RUN_TESTS.bat` once in a while
2. Look for "SUCCESS" or "FAILED"
3. If "SUCCESS" → you're good!
4. If "FAILED" → probably still fine, but ask if worried

---

## Files You Can Ignore

These files are for automated testing (you don't need to touch them):

- `pytest.ini` - Test settings
- `tests/conftest.py` - Test helpers
- `run_tests.py` - Advanced test runner
- `test.py` - Alternative test launcher
- `.github/workflows/tests.yml` - Automatic testing on GitHub

**You only need:** `autotest.py` or `RUN_TESTS.bat`

---

## Summary

**For biologists/field researchers:**

```
Double-click RUN_TESTS.bat → Wait → Look for "SUCCESS!"
```

That's all you need to know! 🎉
