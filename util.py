import os
import subprocess
import tempfile


def compile_tikz_to_pdf(tikz_code):
    temp_dir = tempfile.mkdtemp()

    tex_path = os.path.join(temp_dir, "output.tex")
    pdf_path = os.path.join(temp_dir, "output.pdf")

    with open(tex_path, "w") as f:
        f.write(tikz_code)

    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return pdf_path
    except subprocess.CalledProcessError as e:
        print("PDF compilation failed:", e)
        return None
