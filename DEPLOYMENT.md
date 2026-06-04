# Deployment navodila

Ta datoteka opisuje korake, ki jih je treba narediti v GitHub/DagsHub/Render
racunih. Teh nastavitev ni mogoce izvesti samo lokalno iz kode.

## 1. DVC in DagsHub

1. Namesti DVC z S3 podporo:

   ```powershell
   pipx install "dvc[s3]"
   ```

2. Preveri, da `.dvc/config` kaze na tvoj DagsHub remote:

   ```powershell
   dvc remote list
   ```

3. V GitHub repozitoriju odpri `Settings` > `Secrets and variables` > `Actions`
   in dodaj:

   - `DAGSHUB_ACCESS_KEY_ID`
   - `DAGSHUB_SECRET_ACCESS_KEY`

4. Lokalno lahko celoten pipeline pozenes z:

   ```powershell
   dvc repro
   dvc push
   ```

5. Commitaj DVC definicije:

   ```powershell
   git add dvc.yaml dvc.lock .dvc/config .gitignore
   git commit -m "Add DVC weather pipeline"
   ```

## 2. GitHub Container Registry

Workflow `.github/workflows/docker.yml` objavi Docker sliko v GitHub Container
Registry:

```text
ghcr.io/<uporabnik>/<repo>:latest
ghcr.io/<uporabnik>/<repo>:sha-<commit>
```

Po pushu na `main` ali `master` preveri:

1. GitHub repo > `Actions` > `Docker build and deploy`.
2. GitHub repo > `Packages`, kjer mora biti vidna nova Docker slika.

## 3. Render deploy hook

Ena preprosta produkcijska namestitev je Render Web Service.

1. Na Renderju ustvari nov `Web Service`.
2. Izberi `Existing image` oziroma Docker image iz GHCR.
3. Nastavi image, na primer:

   ```text
   ghcr.io/<uporabnik>/<repo>:latest
   ```

4. Nastavi port `8000`.
5. V Renderju ustvari deploy hook URL.
6. V GitHub repozitoriju dodaj secret:

   ```text
   RENDER_DEPLOY_HOOK_URL=<Render deploy hook URL>
   ```

7. Ponovno zazeni GitHub Actions workflow `Docker build and deploy`.

## 4. Preverjanje produkcije

Ko je deploy koncan, preveri:

```powershell
Invoke-WebRequest https://<tvoj-render-url>/health
```

Pricakovan odgovor:

```json
{"status": "ok", "service": "iis-weather"}
```
