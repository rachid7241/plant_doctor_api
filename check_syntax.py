import ast
import os

def check_python_syntax(filepath):
    """Vérifie la syntaxe d'un fichier Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✅ {filepath} - Syntaxe correcte")
        return True
    except SyntaxError as e:
        print(f"❌ {filepath} - Erreur de syntaxe: {e}")
        return False

def check_all_files():
    """Vérifie tous les fichiers Python du projet"""
    python_files = []
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    all_valid = True
    for filepath in python_files:
        if not check_python_syntax(filepath):
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    print("🔍 Vérification de la syntaxe Python...")
    if check_all_files():
        print("🎉 Tous les fichiers Python ont une syntaxe correcte!")
    else:
        print("💥 Certains fichiers ont des erreurs de syntaxe!")