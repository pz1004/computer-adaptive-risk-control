import urllib.request
import urllib.parse
import json
import time
import os

titles = [
    ("angelopoulos2025learn", "Learn then test: Calibrating predictive algorithms to achieve risk control"),
    ("bates2021distributionfree", "Distribution-free, Risk-controlling Prediction Sets"),
    ("angelopoulos2024conformal", "Conformal Risk Control"),
    ("hoeffding1963probability", "Probability Inequalities for Sums of Bounded Random Variables"),
    ("bentkus2004hoeffding", "On Hoeffding's inequalities"),
    ("maurer2009empirical", "Empirical Bernstein Bounds and Sample Variance Penalization"),
    ("holm1979simple", "A Simple Sequentially Rejective Multiple Test Procedure"),
    ("wiens2003fixed", "A fixed sequence Bonferroni procedure for testing multiple endpoints"),
    ("bretz2008graphical", "A graphical approach to sequentially rejective multiple test procedures"),
    ("tibshirani2020conformal", "Conformal Prediction Under Covariate Shift"),
    ("jin2023sensitivity", "Sensitivity analysis of individual treatment effects: A robust conformal inference approach"),
    ("teerapittayanon2016branchynet", "BranchyNet: Fast inference via early exiting from deep neural networks"),
    ("huang2018multiscale", "Multi-Scale Dense Networks for Resource Efficient Image Classification"),
    ("wang2018skipnet", "SkipNet: Learning Dynamic Routing in Convolutional Networks"),
    ("zhou2020bert", "BERT Loses Patience: Fast and Robust Inference with Early Exit"),
    ("chen2023frugalgpt", "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"),
    ("geifman2017selective", "Selective Classification for Deep Neural Networks"),
    ("vovk2022algorithmic", "Algorithmic Learning in a Random World"),
    ("lei2018distributionfree", "Distribution-Free Predictive Inference for Regression"),
    ("han2022dynamic", "Dynamic Neural Networks: A Survey"),
    ("scardapane2020early", "Why Should We Add Early Exits to Neural Networks?"),
    ("angelopoulos2023gentle", "Conformal Prediction: A Gentle Introduction")
]

os.makedirs('paper/bibitems', exist_ok=True)

for key, title in titles:
    try:
        url = "https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(title) + "&rows=1&select=DOI,title"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if data['message']['items']:
                doi = data['message']['items'][0]['DOI']
                print(f"Found DOI {doi} for {title}")
                
                bib_url = f"https://doi.org/{doi}"
                bib_req = urllib.request.Request(bib_url, headers={'Accept': 'application/x-bibtex', 'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(bib_req) as bib_res:
                    bibtex = bib_res.read().decode('utf-8')
                    with open(f"paper/bibitems/{key}.bib", "w") as f:
                        f.write(bibtex)
            else:
                print(f"Could not find DOI for {title}")
    except Exception as e:
        print(f"Error for {title}: {e}")
    time.sleep(1)
