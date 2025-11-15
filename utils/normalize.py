import re
import unidecode

def normalizar(texto):
    texto = unidecode.unidecode(texto.lower().strip())  # minusculo + remove acentos
    texto = re.sub(r"\s+", "-", texto)                  # substitui espaços por hífen
    texto = re.sub(r"[^a-z0-9\-]", "", texto)           # remove tudo que não for letra, número ou hífen
    return texto