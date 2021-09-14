import os

# UTILS FILE
def rename_gf_folder(name):
    if os.path.exists("generated_files"):
        os.rename("generated_files", name)
        os.mkdir("generated_files")


if __name__ == "__main__":
    pass
