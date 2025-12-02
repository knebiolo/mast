# PyMAST Release Checklist

This checklist ensures consistent, high-quality releases of PyMAST.

---

## Pre-Release (2-3 weeks before)

### Code Preparation

- [ ] All planned features merged to `main` branch
- [ ] No open critical/blocker bugs
- [ ] Code review completed for all new features
- [ ] Deprecated features properly marked with warnings
- [ ] Version number updated in:
  - [ ] `setup.py`
  - [ ] `pyproject.toml`
  - [ ] `pymast/__init__.py` (`__version__`)
  - [ ] `README.md` (if referenced)

### Testing

- [ ] All unit tests passing (`pytest tests/`)
- [ ] Integration tests passing
- [ ] Manual testing of key workflows:
  - [ ] Data import (SRX, Orion, ARES, VR2)
  - [ ] Naive Bayes classification
  - [ ] Bout detection
  - [ ] Overlap resolution
  - [ ] Statistical exports (CJS, LRDR, TTE)
- [ ] Test on all supported platforms:
  - [ ] Windows 10/11
  - [ ] macOS (Intel)
  - [ ] macOS (Apple Silicon)
  - [ ] Linux (Ubuntu)
- [ ] Python version testing:
  - [ ] Python 3.8
  - [ ] Python 3.9
  - [ ] Python 3.10
  - [ ] Python 3.11
  - [ ] Python 3.12

### Documentation

- [ ] CHANGELOG.md updated with all changes
- [ ] Release notes drafted (RELEASE_NOTES_vX.X.X.md)
- [ ] README.md reviewed and updated
- [ ] API documentation updated
- [ ] Tutorial examples tested and working
- [ ] FAQ.md reviewed for new common questions
- [ ] Docstrings complete for all new functions
- [ ] Migration guide written (if breaking changes)

---

## Release Week

### Final Checks

- [ ] Version number confirmed across all files
- [ ] Git status clean (no uncommitted changes)
- [ ] All tests passing on CI/CD
- [ ] No hardcoded paths or credentials in code
- [ ] .gitignore properly excludes test data/outputs
- [ ] License file up to date

### Package Building

- [ ] Clean build directories:
  ```bash
  rm -rf build/ dist/ *.egg-info
  ```
- [ ] Build source distribution:
  ```bash
  python -m build --sdist
  ```
- [ ] Build wheel:
  ```bash
  python -m build --wheel
  ```
- [ ] Test installation from wheel:
  ```bash
  pip install dist/pymast-X.X.X-py3-none-any.whl
  ```
- [ ] Verify package contents:
  ```bash
  tar -tzf dist/pymast-X.X.X.tar.gz | head -20
  ```
- [ ] Check package metadata:
  ```bash
  python -m twine check dist/*
  ```

### Git & GitHub

- [ ] Create release branch: `git checkout -b release/vX.X.X`
- [ ] Commit final version changes
- [ ] Push to GitHub: `git push origin release/vX.X.X`
- [ ] Create Pull Request to main
- [ ] Wait for CI/CD checks to pass
- [ ] Merge to main
- [ ] Tag release: `git tag -a vX.X.X -m "Release vX.X.X"`
- [ ] Push tag: `git push origin vX.X.X`

---

## Release Day

### PyPI Upload

- [ ] Upload to TestPyPI first:
  ```bash
  python -m twine upload --repository testpypi dist/*
  ```
- [ ] Test installation from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ pymast==X.X.X
  ```
- [ ] Run quick smoke test with TestPyPI version
- [ ] Upload to production PyPI:
  ```bash
  python -m twine upload dist/*
  ```
- [ ] Verify on PyPI: https://pypi.org/project/pymast/
- [ ] Test installation from PyPI:
  ```bash
  pip install pymast==X.X.X
  ```

### GitHub Release

- [ ] Go to https://github.com/knebiolo/mast/releases
- [ ] Click "Draft a new release"
- [ ] Select tag: vX.X.X
- [ ] Release title: "PyMAST vX.X.X - [Release Name]"
- [ ] Copy release notes from RELEASE_NOTES_vX.X.X.md
- [ ] Attach distribution files:
  - [ ] Source tarball (.tar.gz)
  - [ ] Wheel (.whl)
- [ ] Check "Create a discussion for this release"
- [ ] Publish release

### Zenodo DOI

- [ ] Verify Zenodo captured GitHub release automatically
- [ ] Visit Zenodo record: https://zenodo.org/record/XXXXX
- [ ] Check metadata is correct:
  - [ ] Authors
  - [ ] License
  - [ ] Keywords
  - [ ] Description
- [ ] Copy DOI badge markdown
- [ ] Update README.md with DOI badge
- [ ] Update PUBLICATIONS.md with new DOI
- [ ] Update RELEASE_NOTES with DOI

---

## Post-Release (Same Day)

### Documentation Updates

- [ ] Update README.md badges:
  - [ ] PyPI version
  - [ ] Zenodo DOI
  - [ ] Download count
- [ ] Update installation instructions with new version
- [ ] Update CHANGELOG.md "Unreleased" section for next version
- [ ] Commit documentation updates
- [ ] Push to GitHub

### Communications

- [ ] Email announcement to users list (if exists)
- [ ] Post on GitHub Discussions
- [ ] Update project website (if exists)
- [ ] Tweet/social media announcement (optional)
- [ ] Notify collaborators/beta testers
- [ ] Update relevant Stack Overflow answers

### Verification

- [ ] Fresh install test on clean machine:
  ```bash
  pip install pymast==X.X.X
  python -c "import pymast; print(pymast.__version__)"
  ```
- [ ] Run example scripts from docs
- [ ] Check PyPI page renders correctly
- [ ] Verify DOI link works
- [ ] Test GitHub release download links

---

## Week After Release

### Monitoring

- [ ] Monitor GitHub Issues for bug reports
- [ ] Monitor PyPI download statistics
- [ ] Check for installation issues in community
- [ ] Review CI/CD logs for any failures
- [ ] Check compatibility reports from users

### Hotfix Planning (if needed)

- [ ] Triage critical bugs
- [ ] Plan vX.X.X+1 hotfix if necessary
- [ ] Update known issues in documentation

### Next Version Planning

- [ ] Create vX.X+1.0 milestone on GitHub
- [ ] Move unfinished features to next milestone
- [ ] Update roadmap based on feedback
- [ ] Schedule next release date

---

## Breaking Change Releases (Major Versions)

Additional checklist items for vX.0.0 releases:

- [ ] **Migration guide** complete and tested
- [ ] **Deprecation warnings** in previous version (if possible)
- [ ] **Blog post** explaining breaking changes
- [ ] **Extended beta period** (2-4 weeks)
- [ ] **Email notifications** to known users
- [ ] **Version comparison table** in docs
- [ ] **Backward compatibility layer** (if feasible)

---

## Hotfix Releases (Patch Versions)

Streamlined checklist for vX.X.X+1 critical fixes:

- [ ] Fix verified and tested
- [ ] Only critical bug fixes (no features)
- [ ] Version bump in all files
- [ ] CHANGELOG.md updated
- [ ] Fast-track testing (critical paths only)
- [ ] Same-day release process
- [ ] Mark as "Hotfix" in release notes

---

## Version Numbering (Semantic Versioning)

**X.Y.Z format:**

- **X (Major)**: Breaking changes, major feature overhauls
  - Example: Database format change, API redesign
  
- **Y (Minor)**: New features, backward-compatible
  - Example: New classifier, additional export format
  
- **Z (Patch)**: Bug fixes, performance improvements
  - Example: Fix crash, speed up overlap resolution

**When to increment:**

- Patch (Z): Bug fixes only, no new features
- Minor (Y): New features, no breaking changes
- Major (X): Breaking changes, deprecation removals

---

## Release Schedule Recommendations

### Regular Releases

- **Minor releases**: Every 3-4 months
- **Patch releases**: As needed for critical bugs
- **Major releases**: Every 12-18 months

### Special Cases

- **Security fixes**: Immediate patch release
- **Critical bugs**: Within 1 week
- **Feature requests**: Bundle into next minor release

---

## Tools & Automation

### Required Tools

```bash
pip install build twine pytest pytest-cov
```

### Automated Scripts

Consider creating:

- `scripts/bump_version.py` - Updates version across all files
- `scripts/build_release.sh` - Automated build and checks
- `scripts/test_install.sh` - Clean environment testing

---

## Common Issues & Solutions

### Issue: Wheel build fails
**Solution:** Check MANIFEST.in includes all necessary files

### Issue: TestPyPI can't find dependencies
**Solution:** TestPyPI doesn't have all packages, expected behavior

### Issue: Import fails after install
**Solution:** Check package structure in wheel, verify __init__.py

### Issue: Version mismatch after install
**Solution:** Old version cached, try `pip install --force-reinstall`

---

## Emergency Rollback

If a release has critical issues:

1. **Yank from PyPI** (doesn't delete, marks as bad):
   ```bash
   # Contact PyPI admins or use web interface
   ```

2. **Delete GitHub release** (if within minutes of posting)

3. **Post immediate notice**:
   - GitHub Discussions
   - README warning
   - Email to users

4. **Prepare hotfix** following process above

5. **Post-mortem**: Document what went wrong in release notes

---

## Checklist Complete?

Before publishing:

- [ ] All checkboxes above completed
- [ ] Release notes proofread
- [ ] Version numbers triple-checked
- [ ] Tests passing on CI/CD
- [ ] At least one other person reviewed changes

**If yes to all → Proceed with release! 🚀**

**If no → Address gaps before publishing**

---

*Template version: 1.0*  
*Last updated: November 24, 2025*
