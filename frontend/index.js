const API = "http://127.0.0.1:5000";

async function setupTrackClassification() {
    const form = document.getElementById("frmPredict");
    const res = document.getElementById("resPredict");
    const btn = document.getElementById("btnPredict");

    let fields = [];
    try {
        fields = await fetch(`${API}/predict-schema`).then(r => r.json());
    } catch (err) {
        alert("Erro ao buscar schema: " + err);
        return;
    }

    const hints = {
        "All Time Rank": "Classificação da música com base em sua popularidade histórica.",
        "Track Score": "Pontuação atribuída à faixa com base em vários fatores",
        "Spotify Streams": "Número total de transmissões no Spotify",
        "Spotify Playlist Count": "Número de playlists do Spotify nas quais a música está incluída.",
        "Spotify Playlist Reach": "Alcance da música nas playlists do Spotify",
        "Spotify Popularity": "Pontuação de popularidade da música no Spotify (0-100)",
        "YouTube Views": "Total de visualizações do vídeo oficial da música no YouTube",
        "YouTube Likes": "Total de curtidas no vídeo oficial da música no YouTube",
        "TikTok Posts": "Número de postagens no TikTok com a música",
        "TikTok Likes": "Total de curtidas em postagens do TikTok com a música.",
        "TikTok Views": "Total de visualizações em postagens do TikTok com a música",
        "YouTube Playlist Reach": "Alcance da música nas playlists do YouTube",
        "Apple Music Playlist Count": "Número de playlists do Apple Music em que a música está incluída",
        "AirPlay Spins": "Número de vezes que a música foi tocada no AirPlay",
        "SiriusXM Spins": "Número de vezes que a música foi tocada no SiriusXM",
        "Deezer Playlist Count": "Número de playlists do Deezer em que a música está incluída",
        "Deezer Playlist Reach": "Alcance da música nas playlists do Deezer",
        "Amazon Playlist Count": "Número de playlists da Amazon Music em que a música está incluída",
        "Pandora Streams": "Número total de transmissões no Pandora",
        "Pandora Track Stations": "Número de estações Pandora que apresentam a música",
        "Soundcloud Streams": "Número total de transmissões no Soundcloud",
        "Shazam Counts": "Número total de vezes que a música foi Shazamada",
    };

    fields.forEach(f => {
        const col = document.createElement("div");
        col.className = "col-md-4 col-lg-2";

        const label = document.createElement("label");
        label.className = "form-label";
        label.innerText = f;

        const inp = document.createElement("input");
        inp.name = f;
        inp.type = "number";
        inp.className = "form-control";
        inp.required = true;
        inp.placeholder = "";

        const hint = document.createElement("div");
        hint.className = "form-text text-muted d-none";
        hint.innerText = hints[f] || "";

        inp.addEventListener("focus", () => hint.classList.remove("d-none"));
        inp.addEventListener("blur", () => hint.classList.add("d-none"));

        col.append(label, inp, hint);
        form.append(col);
    });

    btn.disabled = true;

    const inputs = form.querySelectorAll("input");

    inputs.forEach(input => {
        input.addEventListener("input", () => {
            btn.disabled = !form.checkValidity();
        });
    });

    btn.addEventListener("click", async e => {
        e.preventDefault();
        res.innerText = "Aguardando predição…";
        btn.disabled = true;

        const data = Object.fromEntries(new FormData(form));

        try {
            const { classified, error } = await fetch(`${API}/predict-explicit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            }).then(r => r.json());

            if (error) {
                res.innerText = "Erro: " + error;
                res.className = "text-danger";
            } else {
                res.innerText = classified ? "🚨 Contém conteúdo explícito" : "✅ Não contém conteúdo explícito";
                res.className = classified ? "text-danger" : "text-success";
            }
        } catch (err) {
            res.innerText = "Falha na requisição: " + err;
            res.className = "text-warning";
        } finally {
            btn.disabled = false;
        }
    });
}

async function setupPlatformComparision() {
    try {
        const corr = await fetch(`${API}/platform-correlation`).then(r => r.json());
        const keys = Object.keys(corr);
        const z = keys.map(i => Object.values(corr[i]));
        Plotly.newPlot("heatmap", [{
            z, x: keys, y: keys, type: "heatmap", colorscale: "Viridis"
        }], {
            margin: { t: 40, l: 80 },
            title: "Correlação entre Plataformas"
        });
    } catch (err) {
        document.getElementById("heatmap").innerText = "Erro: " + err;
    }
}

async function setupArtistImpact() {
    try {
        const data = await fetch(`${API}/artist-impact`).then(r => r.json());
        const names = data.map(d => d.Artist);
        const tracks = data.map(d => d.num_tracks);
        const streams = data.map(d => d.total_streams);
        const pops = data.map(d => d.avg_popularity);

        Plotly.newPlot("bubble1", [{
            x: tracks, y: streams, text: names, mode: "markers",
            marker: { size: pops.map(v => v * 2), sizemode: "area" }
        }], { title: "#Faixas vs Streams Totais" });

        Plotly.newPlot("bubble2", [{
            x: pops, y: streams, text: names, mode: "markers",
            marker: { size: tracks.map(v => v * 4), color: "green", sizemode: "area" }
        }], { title: "Popularidade Média vs Streams Totais" });
    } catch (err) {
        document.getElementById("bubble1").innerText = "Erro: " + err;
    }
}

setupTrackClassification();
setupPlatformComparision();
setupArtistImpact();