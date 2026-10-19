# Deploying PitWall AI on AWS (EC2 + Docker)

This puts the app on a small AWS server so anyone with the link can open it, e.g. `http://13.233.45.67/`.
It takes about 30–40 minutes the first time. Everything below works from a Windows PC with only a web
browser and Command Prompt.

**Why EC2 and not App Runner?** Streamlit keeps a live WebSocket connection to each browser. AWS App Runner
doesn't support WebSockets, so a Streamlit app there doesn't work properly. A small EC2 server running
Docker is the simplest option that works.

**Cost.** New AWS accounts (since July 2025) choose a *Free plan* with credits (up to $200) for 6 months.
Small instances such as `t3.micro` and `t3.small` are eligible. The server, its disk and its public IP address
use up the credits slowly, and only while they exist. Check the current rules on aws.amazon.com/free, set a
budget alert (step 1), and **stop the server when you don't need it** (step 9).

---

## 1. Create the AWS account and a budget alert

1. Sign up at **aws.amazon.com/free** and choose the **Free plan**.
2. In the top-right corner, pick a Region close to you, e.g. **Asia Pacific (Mumbai) ap-south-1**. Use the
   same Region for every step.
3. Search **Budgets** → **Create budget** → *Use a template* → **Zero spend budget** → enter your email →
   Create. AWS will email you if anything starts to cost money.

## 2. Launch the server

Search **EC2** → **Launch instance**:

| Setting | Choose |
|---|---|
| Name | `pitwall-ai` |
| Image (AMI) | **Amazon Linux 2023** (default) |
| Instance type | **t3.micro** (free-plan eligible; `t3.small` is smoother if your plan allows it) |
| Key pair | **Create new key pair** → name `pitwall-key`, type RSA, format `.pem` → it downloads |
| Network settings | Tick **Allow SSH traffic from: Anywhere** and **Allow HTTP traffic from the internet** |
| Storage | **16 GiB** gp3 |

Press **Launch instance**, open the instance, and copy its **Public IPv4 address**.

> **Why SSH from Anywhere?** GitHub's servers log in over SSH to update the app (step 8), and their addresses
> change all the time. Login still needs your `.pem` key file (Amazon Linux doesn't allow passwords), so keep that
> file private. If you don't want automatic updates, choose **My IP** instead.

## 3. Copy the project to the server

On your PC, in **Command Prompt**, go to the folder that holds `pitwall_ai.zip` and `pitwall-key.pem`:

```bat
icacls pitwall-key.pem /inheritance:r /grant:r "%USERNAME%:R"
scp -i pitwall-key.pem pitwall_ai.zip ec2-user@YOUR_PUBLIC_IP:~
```

Type `yes` if asked. (The first line makes Windows treat the key file as private, which `scp` requires.)

This first copy is only needed once. After that, GitHub updates the server for you (step 8).

## 4. Open a terminal on the server

From Command Prompt, in the folder with the key:

```bat
ssh -i pitwall-key.pem ec2-user@YOUR_PUBLIC_IP
```

Type `yes` the first time. You're in when the line starts with `[ec2-user@ip-...]$`; `exit` leaves. (The browser
option, EC2 → **Connect** → **EC2 Instance Connect**, also works while SSH is allowed from Anywhere.)

## 5. Install Docker (one time)

```bash
sudo dnf install -y unzip
unzip pitwall_ai.zip
cd pitwall_ai
bash deploy/aws/ec2-setup.sh
exit
```

Reconnect (step 4) so the new Docker permissions apply.

## 6. Add your free AI key

```bash
cd pitwall_ai
cp .env.example .env
nano .env
```

Put your Groq key after `GROQ_API_KEY=` (no quotes), then save with **Ctrl+O**, **Enter**, **Ctrl+X**. Without a
key the app still works; the agents use their rules.

## 7. Build and start the app

```bash
bash deploy/aws/deploy.sh
```

The first build takes 3–6 minutes (it installs Python packages and retrains the ML model if needed). When it
prints the address, open **http://YOUR_PUBLIC_IP/** in your browser.

The container restarts by itself if the server reboots. Decision records and races downloaded with FastF1 are
kept in Docker volumes, so they survive restarts and rebuilds. FastF1 is already installed in the server image;
it uses the server's internet connection, which EC2 allows by default.

## 8. Automatic updates from GitHub

After this one-time setup, every `git push` to the `main` branch runs the tests on GitHub and, if they pass,
copies the code to the server and rebuilds the app (about 3–6 minutes). The server's `.env` (your AI key),
decision records and FastF1 downloads are kept. The workflow is `.github/workflows/deploy.yml`.

1. Put the project on GitHub (see "Put the project on GitHub" below).
2. In the repository: **Settings → Secrets and variables → Actions → New repository secret**. Add two secrets:

   | Name | Value |
   |---|---|
   | `EC2_HOST` | the server's Public IPv4 address, e.g. `52.62.188.157` |
   | `EC2_SSH_KEY` | the whole content of `pitwall-key.pem` (open it in Notepad, select all, copy, paste) |

3. Push any change, or open the **Actions** tab → **Test and deploy** → **Run workflow**. A green tick means the
   site is updated.

The server must already be set up once (steps 1–7), including the `.env` file. If you stop and start the server,
its address changes: update the `EC2_HOST` secret too.

### Put the project on GitHub

1. Install **Git for Windows** (git-scm.com) and create an empty repository on github.com (no README, no
   .gitignore). **Private** is safest.
2. In Command Prompt, in the project folder:

   ```bat
   git init
   git add .
   git status
   ```

   Check that `.env`, `.streamlit/secrets.toml`, `.venv` and any `.pem` file are **not** in the list (the
   `.gitignore` file keeps them out). Then:

   ```bat
   git commit -m "PitWall AI"
   git branch -M main
   git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
   git push -u origin main
   ```

   A browser window asks you to sign in to GitHub the first time.
3. Later changes: `git add .`, `git commit -m "what changed"`, `git push`.

## 9. Stop the server when you're not using it

EC2 → select the instance → **Instance state** → **Stop**. **Start** it again before your demo. The public IP
address changes after a stop/start, so copy the new one. (A fixed address is called an *Elastic IP*; it is
billed while allocated, so only add one if you need it.) Delete everything with **Terminate** when the project
is finished.

## Useful commands (on the server, inside `pitwall_ai`)

| What | Command |
|---|---|
| See the app's log | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Stop | `docker compose down` |
| Check it is healthy | `docker ps` (STATUS should say `healthy`) |

## Troubleshooting

- **"compose build requires buildx 0.17.0 or later":** the setup script installs it; if you set up the server
  with an older copy of the project, run `bash deploy/aws/ec2-setup.sh` again, reconnect, and deploy.
- **The GitHub workflow fails at "Set up the SSH key" or "Copy the code" (timeout):** SSH must be allowed from
  Anywhere in the security group, the server must be running, and `EC2_HOST` must be its current address.
- **"Permission denied (publickey)" in the workflow:** `EC2_SSH_KEY` must be the full `.pem` content, including the
  `-----BEGIN` and `-----END` lines.

- **The page doesn't load:** check the instance's Security group allows **HTTP port 80** from `0.0.0.0/0`,
  and that you used `http://` (not `https://`).
- **Build stops or the server freezes on t3.micro:** the setup script adds 1 GB of swap; if it still struggles,
  stop the instance, change the type to `t3.small` (Actions → Instance settings → Change instance type), start it.
- **"AI model unavailable" in the activity log:** check the key in `.env`, then run `bash deploy/aws/deploy.sh`
  again. On the free Groq plan, many analyses in one minute can hit the rate limit; the rules take over
  automatically.
- **Want only yourself or your teacher to open it?** In the security group, change the HTTP rule's source from
  *Anywhere* to *My IP* (or your teacher's IP). Anyone who can open the page can use your Groq quota (but can't
  see the key).

## Running the same container on your own PC (optional)

With Docker Desktop installed: `docker compose up -d --build` in the project folder, then open
http://localhost/.
