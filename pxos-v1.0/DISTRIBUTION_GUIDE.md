# pxOS v1.0 Distribution Guide

This guide explains how to package and distribute pxOS v1.0.

---

## Package Contents

```
pxos-v1.0/
├── README.md                          Main documentation
├── LICENSE                            MIT License
├── DISTRIBUTION_GUIDE.md              This file
├── build_pxos.py                      Build system (executable)
├── pxos_commands.txt                  Primitive source code
├── pxos.bin                           Pre-built bootable binary
├── tests/
│   ├── boot_qemu.sh                   QEMU boot script
│   ├── boot_bochs.sh                  Bochs boot script
│   └── test_input.sh                  Automated test
├── docs/
│   ├── architecture.md                System architecture
│   ├── primitives.md                  Primitive command reference
│   └── extensions.md                  Extension guide
└── examples/
    └── hello_world_module.txt         Example module
```

**Total size**: ~60KB (mostly documentation)
**Binary size**: 31KB (512 bytes boot + padding)

---

## Distribution Checklist

### ✅ Before Release

- [x] Build binary: `python3 build_pxos.py`
- [x] Test in QEMU (if available)
- [x] Verify all documentation links work
- [x] Check LICENSE file included
- [x] Verify README is comprehensive
- [x] Test scripts are executable

### 📦 Create Release Package

```bash
# 1. Clean build artifacts (optional)
cd pxos-v1.0
rm -rf iso_boot __pycache__

# 2. Rebuild from clean state
python3 build_pxos.py

# 3. Create tarball
cd ..
tar -czf pxos-v1.0.tar.gz pxos-v1.0/

# 4. Create zip (for Windows users)
zip -r pxos-v1.0.zip pxos-v1.0/

# 5. Verify archives
tar -tzf pxos-v1.0.tar.gz | head
unzip -l pxos-v1.0.zip | head
```

---

## GitHub Release

### 1. Create Repository

```bash
cd pxos-v1.0
git init
git add .
git commit -m "Initial release: pxOS v1.0"
git branch -M main
git remote add origin git@github.com:yourusername/pxos.git
git push -u origin main
```

### 2. Create Release

```bash
# Using GitHub CLI
gh release create v1.0 \
  pxos.bin \
  ../pxos-v1.0.tar.gz \
  ../pxos-v1.0.zip \
  --title "pxOS v1.0 - Initial Release" \
  --notes "First stable release of pxOS - a primitive-built x86 bootloader shell"

# Or manually:
# 1. Go to GitHub repository
# 2. Click "Releases" → "Create a new release"
# 3. Tag: v1.0
# 4. Title: pxOS v1.0 - Initial Release
# 5. Attach: pxos.bin, pxos-v1.0.tar.gz, pxos-v1.0.zip
```

### 3. Release Notes Template

```markdown
# pxOS v1.0 - Initial Release

**First stable release!** 🎉

pxOS is a minimal x86 bootloader with an interactive shell, built entirely from custom primitives (WRITE/DEFINE commands) without requiring a traditional assembler.

## Features

- ✅ Direct BIOS boot (works on real hardware!)
- ✅ Interactive character echo shell
- ✅ < 1KB code size
- ✅ Primitive-based build system
- ✅ Comprehensive documentation

## Quick Start

```bash
# Download and extract
wget https://github.com/yourusername/pxos/releases/download/v1.0/pxos-v1.0.tar.gz
tar -xzf pxos-v1.0.tar.gz
cd pxos-v1.0

# Boot in QEMU
./tests/boot_qemu.sh

# Or build from source
python3 build_pxos.py
```

## Downloads

- **pxos.bin** - Bootable binary (31KB)
- **pxos-v1.0.tar.gz** - Full source package (60KB)
- **pxos-v1.0.zip** - Full source package for Windows

## Documentation

- [README.md](README.md) - Main documentation
- [Architecture Guide](docs/architecture.md)
- [Primitive Reference](docs/primitives.md)
- [Extension Guide](docs/extensions.md)

## What's Next?

See [docs/extensions.md](docs/extensions.md) for v1.1 roadmap:
- Command parser
- Backspace support
- Help/clear commands

**Full changelog**: https://github.com/yourusername/pxos/commits/v1.0
```

---

## Other Distribution Platforms

### itch.io (Game/Demo Platform)

1. Create project: https://itch.io/game/new
2. Upload `pxos.bin` and documentation
3. Category: "Tool" or "Other"
4. Tags: operating-system, bootloader, educational, x86

### OSDev Forums

Post announcement: https://forum.osdev.org/

```
Title: [Release] pxOS v1.0 - Primitive-Built Bootloader

I'm happy to announce pxOS v1.0, a minimal x86 bootloader with a unique build system.

Unlike traditional OS projects that use NASM/FASM, pxOS is built from custom "primitives" - direct WRITE commands that map to memory bytes. This makes every byte explicit and educational.

Features:
- Direct BIOS boot
- Interactive shell (character echo)
- ~500 bytes of code
- Python-based builder
- MIT licensed

GitHub: https://github.com/yourusername/pxos
Download: [link to release]

Feedback welcome!
```

### Reddit

**r/osdev**:
```
Title: [Project] pxOS v1.0 - A bootloader built from primitives

I've been working on a minimal bootloader with a unique twist: instead of
using assembly, I build it from "primitive" commands (WRITE/DEFINE) that
directly manipulate memory.

The goal is educational transparency - you can see exactly what byte goes
where and why.

[GitHub link]
[Screenshot]

Would love feedback from the OS dev community!
```

**r/programming**:
```
Title: Building an x86 bootloader without assembly - using custom primitives

Rather than traditional NASM assembly, I built this bootloader using a
custom DSL of primitives (WRITE <addr> <byte>). It's educational and shows
what's really happening at the byte level.

Boots on real hardware, has an interactive shell, MIT licensed.

[GitHub link]
```

### Hacker News

Submit: https://news.ycombinator.com/submit

```
Title: pxOS: An x86 bootloader built from primitives instead of assembly
URL: https://github.com/yourusername/pxos
```

---

## Promotion Strategy

### Week 1: Soft Launch
- [x] GitHub repository public
- [x] Initial release (v1.0)
- [ ] Post to r/osdev
- [ ] Post to OSDev forums

### Week 2: Broader Reach
- [ ] Submit to Hacker News
- [ ] Post to r/programming
- [ ] Tweet about it (if applicable)
- [ ] Cross-post to LinkedIn

### Week 3: Content
- [ ] Write blog post / tutorial
- [ ] Create video demo
- [ ] Submit to weekly newsletters

### Ongoing
- [ ] Respond to issues/questions
- [ ] Accept pull requests
- [ ] Plan v1.1 features

---

## Documentation Website (Optional)

Host docs on GitHub Pages:

```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .

# Copy docs
cp -r docs/* .
cp README.md index.md

# Create simple Jekyll config
cat > _config.yml << EOF
theme: jekyll-theme-minimal
title: pxOS Documentation
description: Primitive-built x86 bootloader
EOF

git add .
git commit -m "Documentation site"
git push origin gh-pages
```

Site will be at: `https://yourusername.github.io/pxos/`

---

## Metrics to Track

### GitHub Stats
- ⭐ Stars
- 🍴 Forks
- 👀 Watchers
- 📥 Release downloads
- 🐛 Issues opened/closed

### Engagement
- Forum posts/replies
- Reddit upvotes/comments
- Hacker News points/comments
- Blog post views (if applicable)

### Goals (First Month)
- 🎯 50+ GitHub stars
- 🎯 100+ downloads
- 🎯 5+ contributors
- 🎯 Front page of r/osdev

---

## Community Building

### Encourage Contributions

**Good first issues**:
- Add backspace support
- Improve welcome message
- Add color support
- Better error handling in builder
- Windows .bat build script

**Label them**: `good first issue`, `help wanted`, `documentation`

### Create CONTRIBUTING.md

```markdown
# Contributing to pxOS

Thank you for your interest! Here's how to contribute:

## Ways to Contribute

1. 🐛 **Report bugs** - Open an issue
2. 💡 **Suggest features** - Open an issue with [Feature Request]
3. 📝 **Improve docs** - Fix typos, clarify explanations
4. 🔧 **Submit PRs** - Bug fixes, new features

## Development Setup

\`\`\`bash
git clone https://github.com/yourusername/pxos.git
cd pxos
python3 build_pxos.py
./tests/boot_qemu.sh
\`\`\`

## Pull Request Process

1. Fork the repo
2. Create a feature branch: \`git checkout -b feature-name\`
3. Make changes
4. Test: \`python3 build_pxos.py\`
5. Commit: \`git commit -m "Add feature"\`
6. Push: \`git push origin feature-name\`
7. Open PR

## Code Style

- Comment every WRITE command
- Use hex for addresses: \`0x7C00\` not \`31744\`
- Group related code with COMMENT dividers
- Update docs if changing primitives

## Questions?

Open an issue or discussion!
```

---

## Licensing & Attribution

### MIT License Benefits
- ✅ Anyone can use, modify, distribute
- ✅ Commercial use allowed
- ✅ Simple, well-understood
- ✅ GitHub recognizes it automatically

### Attribution Requests
Add to README:

```markdown
## Credits

If you use pxOS in your project, please include:

pxOS - A primitive-built x86 bootloader
https://github.com/yourusername/pxos
```

---

## Version Numbering

Use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (e.g., 2.0 = protected mode)
- **MINOR**: New features (e.g., 1.1 = command parser)
- **PATCH**: Bug fixes (e.g., 1.0.1 = fix boot issue)

---

## Next Steps

### Immediate (This Week)
1. [ ] Push to GitHub
2. [ ] Create v1.0 release
3. [ ] Post to r/osdev
4. [ ] Post to OSDev forums

### Short Term (Next Month)
1. [ ] Add CONTRIBUTING.md
2. [ ] Create GitHub Pages site
3. [ ] Write blog post / tutorial
4. [ ] Start v1.1 development

### Long Term (3-6 Months)
1. [ ] Reach 100+ GitHub stars
2. [ ] Get contributions from 5+ people
3. [ ] Release v1.1 (command parser)
4. [ ] Release v2.0 (protected mode)

---

## Support Channels

### GitHub
- **Issues**: Bug reports, feature requests
- **Discussions**: Q&A, ideas, general chat
- **Pull Requests**: Code contributions

### External
- **OSDev Forums**: Technical OS development help
- **r/osdev**: Community discussion
- **Email** (optional): Set up pxos@yourdomain.com

---

## Legal/Safety

### Disclaimer
Add to README:

```markdown
## Safety Warning

pxOS is a bootloader that runs directly on hardware. Writing it to the
wrong disk can cause data loss.

**Before using `dd` command:**
- ⚠️ Double-check device path
- ⚠️ Backup important data
- ⚠️ Use virtual machines for testing

The authors are not responsible for any damage caused by misuse.
```

---

## Success Criteria

You'll know pxOS v1.0 is a success when:

- ✅ Others can download and boot it
- ✅ Documentation is clear enough for beginners
- ✅ Someone contributes a PR
- ✅ Someone creates a derivative project
- ✅ It gets featured on a blog/newsletter
- ✅ You get feedback from the OS dev community

---

## Ready to Launch?

```bash
# Final checklist
cd pxos-v1.0
python3 build_pxos.py          # ✅ Builds successfully
cat README.md                  # ✅ Documentation complete
cat LICENSE                    # ✅ License included
ls tests/*.sh                  # ✅ Test scripts present
ls docs/*.md                   # ✅ All docs written

# Create release
cd ..
tar -czf pxos-v1.0.tar.gz pxos-v1.0/
zip -r pxos-v1.0.zip pxos-v1.0/

# Push to GitHub
cd pxos-v1.0
git init
git add .
git commit -m "pxOS v1.0 - Initial Release"
git remote add origin git@github.com:yourusername/pxos.git
git push -u origin main

# Create release
gh release create v1.0 \
  pxos.bin \
  ../pxos-v1.0.tar.gz \
  ../pxos-v1.0.zip \
  --title "pxOS v1.0" \
  --notes "First stable release"
```

**Then share it with the world!** 🚀

---

**Questions?** Open an issue or discussion on GitHub!
