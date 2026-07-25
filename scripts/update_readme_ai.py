# NVIDIA NIM API README Auto-Summarizer with Jupyter Notebook Content Parsing
# Model: meta/llama-3.3-70b-instruct
import os
import subprocess
import sys
import json
import urllib.request
import urllib.error
import time
import datetime

def extract_notebook_summary(filepath, max_chars=1500):
    """Extracts titles, markdown headings, and key code structures from a .ipynb file."""
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        extracted = []
        cells = nb.get("cells", [])
        for cell in cells[:12]:  # Inspect top 12 cells (overview, headers, imports, architecture)
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown" and source.strip():
                extracted.append(source.strip())
            elif cell_type == "code" and source.strip():
                # Extract key framework imports, model definitions, and comments
                code_snippets = [
                    line for line in source.splitlines() 
                    if line.strip().startswith(("import ", "from ", "class ", "def ", "#", "model =", "Sequential(", "nn.Module"))
                ]
                if code_snippets:
                    extracted.append("[Code Key Snippet]\n" + "\n".join(code_snippets[:5]))

        full_text = "\n---\n".join(extracted)
        return full_text[:max_chars]
    except Exception as e:
        print(f"Error parsing notebook {filepath}: {e}")
        return ""

def get_git_diff_and_commits():
    try:
        log_cmd = ["git", "log", "-n", "5", "--pretty=format:- %s (%h)"]
        log_output = subprocess.check_output(log_cmd, text=True).strip()
        
        stat_cmd = ["git", "diff", "HEAD~1", "HEAD", "--stat"]
        try:
            stat_output = subprocess.check_output(stat_cmd, text=True).strip()
        except Exception:
            stat_output = ""

        name_cmd = ["git", "diff", "HEAD~1", "HEAD", "--name-status"]
        try:
            name_output = subprocess.check_output(name_cmd, text=True).strip()
        except Exception:
            name_output = ""

        # Extract notebook contents for any modified/added .ipynb files
        notebook_details = []
        for line in name_output.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                status, filepath = parts[0], parts[1]
                if filepath.endswith(".ipynb") and status in ("A", "M"):
                    nb_content = extract_notebook_summary(filepath)
                    if nb_content:
                        notebook_details.append(f"=== Notebook Content: {filepath} ===\n{nb_content}")

        nb_text = "\n\n".join(notebook_details) if notebook_details else "No notebook content parsed."

        content = (
            f"Recent Commits:\n{log_output}\n\n"
            f"Changed Files & Subfolders:\n{name_output}\n\n"
            f"File Stats:\n{stat_output}\n\n"
            f"Parsed Notebook Inner Details:\n{nb_text}"
        )
        return content
    except Exception as e:
        print(f"Error getting git info: {e}")
        return "Minor repository updates and maintenance."

def call_nvidia_nim(content, api_key, retries=3):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an automated README update assistant for a Data Science & Deep Learning repository.
Analyze the following recent git commits, changed files, and extracted Jupyter Notebook (.ipynb) inner markdown/code content.
Then write a concise, clean 2-4 bullet point summary describing the actual topics, models, libraries, and practical tasks implemented in the repo.

Format requirements:
- If Jupyter notebooks (.ipynb) or practical folders were added/updated, describe their actual inner domain topic, algorithms, and key libraries used (e.g. Perceptron, PyTorch FNN, Keras ANN, XGBoost, etc.).
- Return ONLY markdown bullet points starting with *.
- Do not include introductory text, conversational filler, or markdown code block fences (no ``` markdown).

Git & Notebook Details:
{content}
"""

    payload = {
        "model": "meta/llama-3.3-70b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    req_data = json.dumps(payload).encode('utf-8')

    for attempt in range(1, retries + 1):
        try:
            print(f"Calling NVIDIA NIM API (attempt {attempt}/{retries})...")
            req = urllib.request.Request(url, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            print(f"Attempt {attempt} HTTP Error {e.code}: {e.reason}\nResponse: {err_body}")
            if attempt < retries:
                time.sleep(3)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Attempt {attempt} Network Error: {e}")
            if attempt < retries:
                time.sleep(3)
        except Exception as e:
            print(f"Attempt {attempt} Unexpected Error: {e}")
            if attempt < retries:
                time.sleep(3)
                
    print("All retries exhausted.")
    return None

def update_readme(summary_text):
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print("README.md not found.")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    start_tag = "<!-- RECENT_UPDATES_START -->"
    end_tag = "<!-- RECENT_UPDATES_END -->"

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    
    formatted_lines = []
    for line in summary_text.splitlines():
        line_str = line.strip()
        if line_str:
            if not line_str.startswith(">"):
                formatted_lines.append(f"> {line_str}")
            else:
                formatted_lines.append(line_str)

    bullets = "\n".join(formatted_lines)
    formatted_block = f"{start_tag}\n> 💡 **Latest Repo Updates ({today}):**\n{bullets}\n{end_tag}"

    if start_tag in readme_content and end_tag in readme_content:
        pre = readme_content.split(start_tag)[0]
        post = readme_content.split(end_tag)[1]
        new_content = pre + formatted_block + post
    else:
        if "## 📌 Projects & Notebooks Overview" in readme_content:
            new_content = readme_content.replace("## 📌 Projects & Notebooks Overview", f"{formatted_block}\n\n## 📌 Projects & Notebooks Overview")
        else:
            new_content = readme_content + f"\n\n{formatted_block}\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("README.md updated successfully!")

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY environment variable not set. Please add NVIDIA_API_KEY to GitHub Secrets.")
        sys.exit(1)

    git_info = get_git_diff_and_commits()
    summary = call_nvidia_nim(git_info, api_key)
    if summary:
        print(f"Generated Summary:\n{summary}\n")
        update_readme(summary)
    else:
        print("ERROR: Failed to generate summary from NVIDIA NIM API.")
        sys.exit(1)

if __name__ == "__main__":
    main()
