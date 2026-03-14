# Secrets Management with SOPS + age

SOPS (Secrets OPerationS) encrypts `.env.secrets` in-place so the file can be committed to Git safely. Only the **values** are encrypted; keys remain visible, making diffs readable. **age** is the encryption backend — modern, fast, and key-pair based (no GPG complexity).

This guide covers the tooling and workflows that are common to all projects using this approach. For project-specific configuration (deploy scripts, app integration), see the documentation in the respective repository.

## Table of Contents

1. [How it works](#how-it-works)
2. [Install SOPS and age](#install-sops-and-age)
   - [Windows (developer machine)](#windows-developer-machine)
   - [Ubuntu server](#ubuntu-server)
3. [First-time setup](#first-time-setup)
   - [Generate an age key pair](#generate-an-age-key-pair)
   - [Register keys in .sops.yaml](#register-keys-in-sopsyaml)
   - [Encrypt .env.secrets for the first time](#encrypt-envsecrets-for-the-first-time)
4. [Daily operations](#daily-operations)
   - [Edit secrets](#edit-secrets)
   - [Decrypt locally](#decrypt-locally)
5. [Team management](#team-management)
   - [Add a developer](#add-a-developer)
   - [Remove a developer](#remove-a-developer)
6. [CI/CD integration](#cicd-integration)
7. [Troubleshooting](#troubleshooting)

---

## How it works

```
.env.secrets (plaintext, gitignored)
        │
        │  sops --encrypt
        ▼
.env.secrets.enc (encrypted, committed to Git)
        │
        │  sops --decrypt  (on deployment or local setup)
        ▼
.env.secrets (plaintext, runtime only)
        │
        │  loaded by the application
        ▼
Application process (secrets in memory)
```

An encrypted file looks like this in Git:

```dotenv
SECRET_KEY=ENC[AES256_GCM,data:xK3pAbc...,iv:...,tag:...,type:str]
DB_PASSWORD=ENC[AES256_GCM,data:mN8qDef...,iv:...,tag:...,type:str]
STRIPE_SECRET_KEY=ENC[AES256_GCM,data:pQ7rGhi...,iv:...,tag:...,type:str]
```

The key names are visible; only the values are encrypted.

---

## Install SOPS and age

### Windows (developer machine)

**Option A — winget (recommended)**

```powershell
winget install FiloSottile.age
winget install mozilla.sops
```

**Option B — Scoop**

```powershell
scoop install age
scoop install sops
```

**Option C — Manual download**

1. Download `age` from <https://github.com/FiloSottile/age/releases/latest>
   - Pick `age-v1.x.x-windows-amd64.zip`
   - Extract `age.exe` and `age-keygen.exe` to a folder on your `PATH` (e.g. `C:\tools\`)

2. Download `sops` from <https://github.com/getsops/sops/releases/latest>
   - Pick `sops-v3.x.x.exe`
   - Rename to `sops.exe` and place in the same folder

**Verify installation**

```powershell
age --version
sops --version
```

---

### Ubuntu server

```bash
# age — available in Ubuntu 22.04+ standard repos
sudo apt update && sudo apt install -y age

# sops — install the latest release binary
SOPS_VERSION=$(curl -s https://api.github.com/repos/getsops/sops/releases/latest \
    | grep '"tag_name"' | cut -d'"' -f4)
curl -Lo /tmp/sops "https://github.com/getsops/sops/releases/download/${SOPS_VERSION}/sops-${SOPS_VERSION}.linux.amd64"
sudo mv /tmp/sops /usr/local/bin/sops
sudo chmod +x /usr/local/bin/sops

# Verify
age --version
sops --version
```

---

## First-time setup

### Generate an age key pair

Run this **once per machine** (developer laptop or server). Each machine gets its own key pair.

**Windows (PowerShell)**

```powershell
# Create the default SOPS key directory
New-Item -ItemType Directory -Force "$env:APPDATA\sops\age" | Out-Null

# Generate the key pair
age-keygen -o "$env:APPDATA\sops\age\keys.txt"
# Output: Public key: age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p
```

**Ubuntu / Linux**

```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
# Output: Public key: age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p
```

**Production server** — store the key outside the home directory, readable only by root:

```bash
sudo mkdir -p /etc/sops/age
sudo age-keygen -o /etc/sops/age/keys.txt
sudo chmod 600 /etc/sops/age/keys.txt
# Note down the public key — you will add it to .sops.yaml
```

> **Keep the private key safe.** The `.txt` file contains both the private and public key.
> Never commit it to Git. Back it up in a password manager or secure vault.

---

### Register keys in .sops.yaml

Open `.sops.yaml` at the project root and replace the placeholder values with the real public keys:

```yaml
creation_rules:
  - path_regex: \.env\.secrets(\..*)?$
    age: >-
      age1<developer-1-public-key>,
      age1<developer-2-public-key>,
      age1<prod-server-public-key>
```

Add one key per line, separated by commas. Every key listed here can decrypt the secrets.

Commit this file to Git — it only contains **public keys**, which are not sensitive.

---

### Encrypt .env.secrets for the first time

Make sure your `.env.secrets` is complete and correct before encrypting.

> **SOPS 3.7.x note:** The dotenv parser in SOPS ≤ 3.7.x rejects comment lines (`#`) and blank lines. Strip them first using the commands below. SOPS 3.8+ handles this transparently.

**Linux / macOS**

```bash
# Strip comments and blank lines, then encrypt
grep -v '^\s*#' .env.secrets | grep -v '^\s*$' > .env.secrets.nocomments
sops --encrypt --input-type dotenv --output-type dotenv .env.secrets.nocomments > .env.secrets.enc
rm .env.secrets.nocomments

# Verify the output looks correct (values should be ENC[...] blobs)
head -5 .env.secrets.enc
```

**Windows (PowerShell)**

```powershell
# Strip comments and blank lines, then encrypt
(Get-Content .env.secrets | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '\S' }) |
    Set-Content .env.secrets.nocomments
sops --encrypt --input-type dotenv --output-type dotenv .env.secrets.nocomments > .env.secrets.enc
Remove-Item .env.secrets.nocomments

# Verify
Get-Content .env.secrets.enc | Select-Object -First 5
```

```bash
# Commit the encrypted file
git add .env.secrets.enc .sops.yaml
git commit -m "chore: add SOPS-encrypted secrets"

# The plaintext .env.secrets remains gitignored and stays only on disk
```

---

## Daily operations

### Edit secrets

The recommended workflow is to use `sops` as an interactive editor — it decrypts, opens your editor, and re-encrypts on save:

```bash
sops .env.secrets.enc
```

SOPS uses the `$EDITOR` environment variable (defaults to `vi` on Linux, `notepad` on Windows).

**Windows — set a preferred editor (PowerShell)**

```powershell
$env:EDITOR = "code --wait"   # VS Code
sops .env.secrets.enc
```

**Linux — set a preferred editor**

```bash
EDITOR="nano" sops .env.secrets.enc
# or permanently:
export EDITOR=nano
```

After you save and close the editor, SOPS re-encrypts the file in place. Commit the updated `.env.secrets.enc`.

---

### Decrypt locally

After a `git pull` that contains a changed `.env.secrets.enc`, decrypt it manually:

**Windows**

```powershell
sops --decrypt --input-type dotenv --output-type dotenv .env.secrets.enc | Set-Content -Encoding UTF8 .env.secrets
```

**Linux / macOS**

```bash
sops --decrypt --input-type dotenv --output-type dotenv .env.secrets.enc > .env.secrets
```

---

## Team management

### Add a developer

1. Ask the new developer to generate an age key pair (see [Generate an age key pair](#generate-an-age-key-pair)) and send you their **public key**.

2. Add the public key to `.sops.yaml`:
   ```yaml
   age: >-
     age1<existing-dev>,
     age1<new-dev-public-key>,
     age1<prod-server>
   ```

3. Re-encrypt the file so the new key can decrypt it:
   ```bash
   # sops rotate re-wraps the data key for all listed recipients — no comment stripping needed
   sops rotate -i .env.secrets.enc
   ```

4. Commit the updated `.sops.yaml` and `.env.secrets.enc`:
   ```bash
   git add .sops.yaml .env.secrets.enc
   git commit -m "chore: add <name> to SOPS secrets access"
   ```

5. The new developer can now decrypt with their private key.

---

### Remove a developer

1. Remove their public key from `.sops.yaml`.

2. **Rotate the encryption key** — this re-encrypts all values with a new data key, making the old key permanently unable to decrypt future versions:
   ```bash
   sops rotate -i .env.secrets.enc
   ```

3. If the developer may have had access to production secrets, also rotate the actual secret values (e.g. generate a new `SECRET_KEY`, `DB_PASSWORD`, etc.), then re-encrypt:
   ```bash
   sops .env.secrets.enc   # edit values
   ```

4. Commit:
   ```bash
   git add .sops.yaml .env.secrets.enc
   git commit -m "chore: revoke <name> SOPS access and rotate secrets"
   ```

---

## CI/CD integration

Store the age private key as a CI secret variable (e.g. `SOPS_AGE_KEY`), then decrypt in the pipeline:

```yaml
# Example: GitHub Actions
- name: Decrypt secrets
  env:
    SOPS_AGE_KEY: ${{ secrets.SOPS_AGE_KEY }}
  run: |
    sops --decrypt --input-type dotenv --output-type dotenv .env.secrets.enc > .env.secrets
```

The CI machine does not need a key file on disk — the raw key string in `SOPS_AGE_KEY` is sufficient.

---

## Troubleshooting

**`failed to load age identities … did not find keys in … /root/.config/sops/age/keys.txt`**

This happens on the server when the age private key is stored at `/etc/sops/age/keys.txt` but SOPS's built-in search only looks in `~/.config/sops/age/keys.txt`.

**Quick fix** — verify the key file exists and has correct permissions:

```bash
sudo ls -la /etc/sops/age/keys.txt   # should be -rw------- root root
# If missing, generate or copy the key:
sudo mkdir -p /etc/sops/age && sudo chmod 700 /etc/sops/age
sudo age-keygen -o /etc/sops/age/keys.txt
sudo chmod 600 /etc/sops/age/keys.txt
# Then add the printed public key to .sops.yaml and re-encrypt on your dev machine:
# sops rotate -i .env.secrets.enc
```

Pass `SOPS_AGE_KEY_FILE` to point SOPS at a non-standard key location:

```bash
SOPS_AGE_KEY_FILE=/path/to/keys.txt sops --decrypt ...
```

> **Note:** Adding `SOPS_AGE_KEY_FILE` to `/etc/environment` does **not** work with `sudo` — sudo drops most environment variables by default. Always pass it inline.
