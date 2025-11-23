# GitHub Desktop Workflow for PyMAST

This guide explains how to merge your feature branch back to main using GitHub Desktop.

## 📋 Pre-Merge Checklist

Before merging, ensure:
- ✅ All code is working and tested
- ✅ Documentation is complete
- ✅ Temporary files are cleaned up
- ✅ No syntax errors (check Problems panel in VS Code)

## 🔄 Step-by-Step Merge Process

### Step 1: Review Your Changes

1. Open **GitHub Desktop**
2. Check current branch (shown at top of window)
3. Review the **Changes** tab - you should see:
   - Modified files in `pymast/` (updated docstrings)
   - New file: `GETTING_STARTED.md`
   - Modified: `README.md`
   - Deleted temporary markdown files

### Step 2: Commit Your Changes

1. In **GitHub Desktop**, bottom-left panel:
   - **Summary** (required): Enter commit message
     ```
     Add comprehensive documentation and clean up repository
     ```
   - **Description** (optional): Add details
     ```
     - Added Google/NumPy docstrings to all modules
     - Created GETTING_STARTED.md for new users
     - Updated README.md with cleaner structure
     - Removed temporary development documentation
     - All modules now support Python help() system
     ```

2. Click **Commit to [your-branch-name]** button

### Step 3: Push to GitHub

1. Click **Push origin** button (top-right)
   - This uploads your commits to GitHub
   - Wait for upload to complete

### Step 4: Switch to Main Branch

1. Click current branch dropdown (top of window)
2. Select **main** from the list
3. GitHub Desktop will switch branches

### Step 5: Merge Your Feature Branch

1. In top menu: **Branch** → **Merge into current branch**
2. Select your feature branch from the list
3. Click **Merge [branch-name] into main**
4. GitHub Desktop will merge the changes

### Step 6: Handle Merge Conflicts (if any)

If conflicts occur:
1. GitHub Desktop will show **Conflicts** warning
2. Click **Open in Visual Studio Code**
3. VS Code will highlight conflicts
4. Resolve each conflict by choosing correct version
5. Save files
6. Return to GitHub Desktop
7. Click **Continue merge**

### Step 7: Push Merged Main Branch

1. After successful merge, click **Push origin**
2. This updates GitHub with your merged main branch

### Step 8: Clean Up (Optional)

1. In branch dropdown, right-click your feature branch
2. Select **Delete...** to remove old branch locally
3. Confirm deletion

## ✅ Verification

After merge, verify:
1. GitHub Desktop shows "main" as current branch
2. No uncommitted changes
3. Recent commit shows your merge in history

## 🆘 Common Issues

### "Cannot merge due to uncommitted changes"
**Solution**: Commit or discard changes before merging

### "Merge conflicts detected"
**Solution**: Follow Step 6 above to resolve conflicts

### "Your branch is behind origin/main"
**Solution**: Click "Pull origin" to get latest changes before merging

### "Push rejected"
**Solution**: Someone else pushed to main. Pull first, then push again.

## 🎯 Quick Reference

```
Current workflow:
1. Commit changes → "Commit to [branch]"
2. Push changes → "Push origin"  
3. Switch to main → Branch dropdown → "main"
4. Merge → Branch → "Merge into current branch"
5. Push main → "Push origin"
6. Delete branch → Right-click branch → "Delete"
```

## 💡 Best Practices

- **Commit often** - Small, focused commits are easier to review
- **Descriptive messages** - Future you will thank you
- **Pull before push** - Always sync before pushing
- **Test before merge** - Ensure code works on feature branch first
- **Review changes** - Check diff before committing

## 📱 GitHub Desktop Buttons

| Button | Action |
|--------|--------|
| **Fetch origin** | Check for remote changes (doesn't download) |
| **Pull origin** | Download and merge remote changes |
| **Push origin** | Upload your commits to GitHub |
| **Current Branch** | Switch between branches |
| **Commit to [branch]** | Save your changes with message |

---

**Need help?** GitHub Desktop has built-in tutorials:
Help menu → GitHub Desktop Help → Search for "merge branches"
