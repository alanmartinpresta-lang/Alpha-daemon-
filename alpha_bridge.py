"""
Passerelle optionnelle ALPHA LAB.

Elle n'est PAS activée par défaut.
Objectif :
1. recevoir une question de l'interface ;
2. exécuter Alpha localement ;
3. éventuellement envoyer le contexte à un modèle via l'API OpenAI ;
4. retourner une réponse structurée à l'interface.

Important : ne jamais mettre une clé API OpenAI dans le JavaScript du navigateur.
La clé doit rester côté serveur dans OPENAI_API_KEY.
"""
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        raw=json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._json(200, {"ok": True})

    def do_POST(self):
        if self.path != "/exchange":
            return self._json(404, {"error":"not found"})
        n=int(self.headers.get("Content-Length","0"))
        body=json.loads(self.rfile.read(n) or b"{}")
        question=body.get("question","").strip()
        # Placeholder volontaire : brancher ici le vrai runtime Alpha.
        # Rien n'est présenté comme une réponse réelle tant que le runtime n'est exécuté.
        response = {
            "alpha_response": "PASSERELLE NON CONFIGURÉE : le serveur a bien reçu la question, mais le runtime Alpha n'est pas branché à cet endpoint.",
            "translation": "Le système a reçu la question mais n'a pas encore obtenu une réponse du runtime Alpha.",
            "interpretation": "Aucune conclusion sur Alpha ne doit être tirée de ce message."
        }
        return self._json(200, response)

if __name__ == "__main__":
    print("ALPHA LAB bridge: http://localhost:8787")
    HTTPServer(("0.0.0.0",8787),Handler).serve_forever()
