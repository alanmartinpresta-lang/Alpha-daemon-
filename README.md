# ALPHA LAB — interface web

Cette archive ajoute une interface web sombre, mobile-first et utilisable dans Safari.

## Ce qui est déjà fonctionnel

- Interface ALPHA LAB inspirée du visuel demandé.
- Bouton **ÉCHANGER AVEC ALPHA**.
- Champ pour poser une question.
- Bibliothèque locale de **2000 questions/réponses de secours**.
- Recherche dans la bibliothèque.
- PWA : installation possible sur iPhone quand le site est servi en HTTPS.
- Fonctionnement sans serveur grâce au fallback local.
- Architecture prête pour une passerelle serveur vers le runtime Alpha.

## Point important

La bibliothèque de 2000 réponses est une **bibliothèque de secours générée**, pas 2000 réponses réellement produites par Alpha.

Le mode réel doit exécuter le runtime Alpha et retourner sa sortie. L'application ne prétend donc jamais qu'une réponse simulée est une réponse réelle.

## Connexion permanente : non

Par défaut, aucune connexion ChatGPT/API n'est activée.

Le fichier `bridge/config.json` contient :

```json
{"enabled": false, "endpoint": "http://localhost:8787/exchange"}
```

On pourra ensuite déployer une passerelle persistante et mettre `enabled` à `true`.

## Passerelle serveur

`bridge/alpha_bridge.py` est un squelette volontairement neutre. Il évite de mettre une clé secrète dans l'application iPhone.

Pour une vraie connexion à un modèle OpenAI, la clé doit rester côté serveur dans `OPENAI_API_KEY`. La documentation officielle actuelle utilise la Responses API.

## Installation dans ton dépôt

Copier le dossier `alpha-lab-webapp/` dans le dépôt Alpha.

Puis servir ce dossier avec un hébergement HTTPS. Pour GitHub Pages, le plus simple est de publier ce dossier comme site statique.

Pour tester localement :

```bash
python3 -m http.server 8080 --directory alpha-lab-webapp
```

Puis ouvrir `http://localhost:8080`.

## Prochaine étape recommandée

Ne pas ajouter de nouvelles fonctions complexes tout de suite.

1. Vérifier l'interface dans Safari.
2. Vérifier la bibliothèque locale.
3. Brancher l'endpoint réel sur le runtime Alpha.
4. Ensuite seulement ajouter la passerelle modèle/ChatGPT côté serveur.
