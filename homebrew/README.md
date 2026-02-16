# Homebrew Tap for vcon-mac-wtf

## Install

```bash
brew tap thomashowe/vcon
brew install vcon-mac-wtf
```

## Adding this formula to your tap

1. Create a repo `homebrew-vcon` with structure:
   ```
   homebrew-vcon/
   Formula/
     vcon-mac-wtf.rb   # copy from this directory
   ```

2. The formula fetches v0.1.0 from PyPI. For new releases, update `url` and `sha256` in the formula (get sha256 via `curl -sL <url> | shasum -a 256`).
