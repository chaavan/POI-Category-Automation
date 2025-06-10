# -*- coding: utf-8 -*-
import streamlit as st
from overturemaps import core
import pandas
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
import google.generativeai as genai
import threading

# --- Global variables for progress tracking (for thread safety with Streamlit) ---
# For website scraping
g_scraped_sites_count = 0
g_total_sites = 0
g_site_progress_bar = None
g_site_progress_text = None
g_site_lock = threading.Lock()

# For social media scraping
g_scraped_socials_count = 0
g_total_social_urls = 0
g_social_progress_bar = None
g_social_progress_text = None
g_social_lock = threading.Lock()

# --- Helper Functions (adapted from your script) ---
def get_first_website(websites_list):
    if websites_list and isinstance(websites_list, list) and len(websites_list) > 0 and websites_list[0] and isinstance(websites_list[0], str):
        return websites_list[0]
    return ""

def scrape_website(url, timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No title found'
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')
        h1_tags = [tag.get_text(strip=True) for tag in soup.find_all('h1')]
        scraped_text = f"Title: {title}. Meta Description: {meta_desc}. Headings: {';'.join(h1_tags)}"
        return scraped_text
    except Exception as e:
        # st.warning(f"Failed to Scrape: {url}: {e}") # Avoid Streamlit calls directly in threads if possible
        print(f"Failed to Scrape: {url}: {e}") # Log to console
        return ""

# --- Category Keywords and Rules (from your script) ---
category_keywords = {
    # Accommodation
    "accommodation-hotel":          ["hotel", "albergo"],
    "accommodation-hostel":         ["hostel", "ostello"],
    "accommodation-motel":          ["motel"],
    "accommodation-resort":         ["resort"],
    "accommodation-bed_and_breakfast": ["bed and breakfast", "b&b"],
    "eat_and_drink-restaurant":     ["restaurant", "ristorante", "trattoria", "osteria", "pizzeria", "pizza"],
    "eat_and_drink-cafe":           ["cafe", "caffè", "coffee shop", "caffetteria"],
    "eat_and_drink-bar":            ["bar", "pub", "wine bar", "enoteca"],
    "retail-food-bakery":           ["bakery", "panificio", "forno"],
    "retail-food-ice_cream_shop":   ["ice cream", "gelateria"],
    "retail-food-pastry_and_cake_shop": ["pastry shop", "pasticceria"],
    "eat_and_drink-fast_food":      ["fast food", "takeaway", "kebab", "shawarma", "sushi", "taco", "burger", "hot dog"],
    "eat_and_drink-tea_room":       ["tea room", "tearoom"],
    "eat_and_drink-juice_bar":      ["juice bar", "smoothie"],
    "retail":                       ["shop", "store", "market", "boutique", "emporio", "negozio"],
    "retail-food-supermarket":      ["supermarket", "grocery", "ipermercato"],
    "health_and_medical-pharmacy":  ["pharmacy", "farmacia"],
    "retail-books_stationery_music_and_film-book_shop": ["bookstore", "libreria", "librairie"],
    "retail-clothing_and_accessories-clothing_store":    ["clothing store", "abbigliamento", "fashion"],
    "retail-clothing_and_accessories-jewelry_and_watch_store": ["jewelry", "gioielleria"],
    "retail-clothing_and_accessories-shoe_store":        ["shoe store", "calzature"],
    "retail-toys_and_games_store":     ["toy store", "giocattoli"],
    "retail-home_and_garden-florist":  ["flower shop", "florist", "fioraio"],
    "retail-electronics-consumer_electronics_store": ["electronics store", "elettronica"],
    "retail-home_and_garden-hardware_store": ["hardware store", "ferramenta"],
    "retail-department_store":         ["department store", "grande magazzino"],
    "retail-shopping_center_and_mall":["mall", "shopping center", "centro commerciale"],
    "retail-beverage_store-wine_and_spirits_store": ["wine shop", "enoteca"],
    "retail-books_stationery_music_and_film-newsagent_and_kiosk": ["newsstand", "edicola"],
    "retail-tobacconist":              ["tobacco shop", "tabaccheria", "tabacchi"],
    "attractions_and_activities-museum":     ["museum", "museo"],
    "arts_and_entertainment-movie_theater": ["cinema", "movie theater"],
    "arts_and_entertainment-performing_arts_theater":["theater", "teatro"],
    "attractions_and_activities-art_gallery": ["art gallery", "galleria d'arte"],
    "arts_and_entertainment-topic_concert_venue":["music venue", "concert hall"],
    "arts_and_entertainment-night_club":    ["nightclub", "discoteca", "disco"],
    "attractions_and_activities-amusement_park":["amusement park", "theme park"],
    "attractions_and_activities-aquarium":   ["aquarium", "acquario"],
    "attractions_and_activities-zoo":        ["zoo", "bioparco"],
    "arts_and_entertainment-bowling_alley":  ["bowling", "bowling alley"],
    "active_life-sports_and_recreation_venue-gym_and_fitness_center": ["gym", "fitness", "palestra"],
    "active_life-sports_and_recreation_venue-stadium_and_arena": ["stadium", "arena", "stadio"],
    "active_life-sports_and_recreation_venue-sports_center": ["sports centre", "centro sportivo"],
    "active_life-sports_and_recreation_venue-public_swimming_pool": ["swimming pool", "piscina"],
    "attractions_and_activities-playground": ["playground", "parco giochi"],
    "active_life-marina":                  ["marina", "porto"],
    "attractions_and_activities-beach":     ["beach", "spiaggia"],
    "attractions_and_activities-historic_site": ["historic site", "monument", "monumento"],
    "attractions_and_activities-winery":    ["winery", "cantina"],
    "attractions_and_activities-brewery":   ["brewery", "birrificio"],
    "arts_and_entertainment-arcade":       ["arcade", "sala giochi"],
    "arts_and_entertainment-casino":       ["casino", "casinò"],
    "arts_and_entertainment-music_school": ["music school", "scuola di musica"],
    "attractions_and_activities-library":   ["library", "biblioteca"],
    "art_studios":                         ["art studio", "studio d'arte"],
    "health_and_medical-hospital":         ["hospital", "ospedale"],
    "health_and_medical-clinic_and_medical_center": ["clinic", "ambulatorio", "poliambulatorio"],
    "health_and_medical-dentist":          ["dentist", "dentista"],
    "health_and_medical-doctor":           ["doctor", "physician", "medico"],
    "pets-veterinarian":                   ["veterinary", "vet", "veterinario"],
    "health_and_medical-optician":         ["optician", "ottica"],
    "health_and_medical-physiotherapist":  ["physio", "physiotherapist", "fisioterapista"],
    # "health_and_medical-pharmacy":  ["pharmacy", "farmacia"], # Duplicate, already above
    "health_and_medical-diagnostic_lab":   ["laboratory", "laboratorio"],
    "health_and_medical-urgent_care":      ["urgent care", "pronto soccorso"],
    "health_and_medical-medical_supply":   ["medical supply", "dispositivi medici"],
    "education-college_university":        ["university", "college", "università", "ateneo", "politecnico"],
    "education-school":                    ["school", "scuola", "liceo", "istituto"],
    "education-specialty_school-driving_school": ["driving school", "autoscuola"],
    "education-school-preschool_and_kindergarten": ["kindergarten", "nursery", "asilo"],
    "public_service_and_government-post_office": ["post office", "ufficio postale", "poste italiane"],
    "public_service_and_government-police_station": ["police", "polizia", "carabinieri", "questura"],
    "public_service_and_government-fire_station": ["fire station", "vigili del fuoco"],
    "public_service_and_government-embassy":["embassy", "ambasciata"],
    "public_service_and_government-consulate":["consulate", "consolato"],
    "public_service_and_government-government_services-city_hall":["city hall", "municipio", "comune"],
    "public_service_and_government-courthouse":["courthouse", "tribunale"],
    "public_service_and_government-library":["library", "biblioteca"],
    "financial_service-bank_credit_union":["bank", "banca", "ATM", "bancomat"],
    "financial_service-insurance_agency":["insurance", "assicurazioni"],
    "real_estate-real_estate_agent_and_broker":["real estate", "agenzia immobiliare"],
    "real_estate-property_management":["property management", "gestione immobiliare"],
    "financial_service-ATM":["ATM", "bancomat"],
    "financial_service-stock_broker":["broker", "borsa"],
    "travel-airport":                      ["airport", "aeroporto"],
    "travel-transportation-rail_station":  ["train station", "stazione ferroviaria"],
    "travel-transportation-bus_station":   ["bus station", "autostazione"],
    "travel-transportation-bus_stop":      ["bus stop", "fermata autobus"],
    "travel-transportation-subway_station":["metro station", "subway station", "stazione metro"],
    "travel-transportation-taxi_limo_and_shuttle_service-taxi_stand":["taxi stand", "posteggio taxi"],
    "automotive-gas_station":             ["gas station", "petrol station", "distributore di benzina"],
    "travel-road_structures_and_services-parking":["parking", "parcheggio", "autorimessa"],
    "automotive-automotive_services_and_repair-car_wash_and_detail":["car wash", "autolavaggio"],
    "automotive-automotive_services_and_repair":["car repair", "mechanic", "officina", "carrozzeria"],
    "automotive-automotive_dealer":["car dealer", "concessionaria auto"],
    "automotive-automotive_parts_and_accessories-tire_shop":["tire shop", "gommista"],
    "beauty_and_spa-hair_salon":["hair salon", "parrucchiere", "barber"],
    "beauty_and_spa-beauty_salon":["beauty salon", "centro estetico", "istituto di bellezza"],
    "professional_services-laundry_services":["laundry", "lavanderia", "launderette"],
    "professional_services-funeral_services_and_cemeteries-funeral_service":["funeral home", "onoranze funebri"],
    "professional_services-funeral_services_and_cemeteries-cemetery":["cemetery", "cimitero"],
    "professional_services-dry_cleaning":["dry cleaning", "lavasecco"],
    "professional_services-pet_grooming":["pet grooming", "toelettatura animali"],
    "professional_services-photographer":["photographer", "fotografo"],
    "professional_services-legal_services":["lawyer", "avvocato", "studio legale"],
    "professional_services-accounting":["accountant", "commercialista"],
    "attractions_and_activities-park":["park", "parco", "giardino pubblico"],
    # "attractions_and_activities-beach":["beach", "spiaggia"], # Duplicate
    "attractions_and_activities-hiking_trail":["trail", "sentiero"],
    "attractions_and_activities-campground":["campground", "campeggio"],
    "attractions_and_activities-ski_resort":["ski resort", "stazione sciistica"],
}

RULES = [
    (
        lambda n, w, s, kws=keywords: any(kw in (n or "").lower() or kw in (w or "").lower() or kw in (s or "").lower()
                                          for kw in kws),
        category
    )
    for category, keywords in category_keywords.items()
]

def rule_based_category(poi_name, website_text, social_text):
    name = (poi_name or "").lower()
    web  = (website_text or "").lower()
    soc  = (social_text or "").lower()
    for test_fn, category in RULES:
        try:
            if test_fn(name, web, soc):
                return category
        except Exception:
            continue
    return None

# --- Worker Functions for Concurrent Scraping (with Streamlit progress update) ---
def scrape_site_wrapper_st(args_tuple):
    global g_scraped_sites_count, g_total_sites, g_site_progress_bar, g_site_progress_text, g_site_lock
    place, site_list = args_tuple # site_list is like place_dataset.websites[i]

    site_url_to_scrape = None
    if site_list and isinstance(site_list, list) and len(site_list) > 0 and site_list[0] and site_list[0] != "No website Found":
        site_url_to_scrape = site_list[0]

    result_content = "No website Found"
    if site_url_to_scrape:
        try:
            scraped_content = scrape_website(site_url_to_scrape)
            result_content = scraped_content if scraped_content else "Scraping Failed"
        except Exception:
            result_content = "Scraping Failed (Exception)"
    
    with g_site_lock:
        g_scraped_sites_count += 1
        if g_total_sites > 0:
            percent = g_scraped_sites_count / g_total_sites
            if g_site_progress_text:
                g_site_progress_text.caption(f"Scraping websites: {g_scraped_sites_count}/{g_total_sites} ({percent*100:.1f}%)")
            if g_site_progress_bar:
                g_site_progress_bar.progress(percent)
    return place, result_content

def scrape_social_url_wrapper_st(args_tuple):
    global g_scraped_socials_count, g_total_social_urls, g_social_progress_bar, g_social_progress_text, g_social_lock
    place_name, url = args_tuple
    scraped_content = scrape_website(url)

    with g_social_lock:
        g_scraped_socials_count += 1
        if g_total_social_urls > 0:
            percent = g_scraped_socials_count / g_total_social_urls
            if g_social_progress_text:
                g_social_progress_text.caption(f"Scraping social URLs: {g_scraped_socials_count}/{g_total_social_urls} ({percent*100:.1f}%)")
            if g_social_progress_bar:
                g_social_progress_bar.progress(percent)
    return place_name, url, scraped_content if scraped_content else "Scraping Failed or No Content"


# --- Main Data Processing Function ---
# @st.cache_data # Caching can be complex with external API calls and progress bars. Use with caution.
def process_poi_data(bbox_tuple, api_key, streamlit_progress_elements):
    global g_scraped_sites_count, g_total_sites, g_site_progress_bar, g_site_progress_text
    global g_scraped_socials_count, g_total_social_urls, g_social_progress_bar, g_social_progress_text

    # Assign Streamlit progress elements to global vars
    (g_site_progress_text, g_site_progress_bar) = streamlit_progress_elements["websites"]
    (g_social_progress_text, g_social_progress_bar) = streamlit_progress_elements["socials"]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro") # Using "gemini-pro" as "gemini-2.0-flash" might not be available or intended
    except Exception as e:
        st.error(f"Failed to configure Google Generative AI: {e}")
        return None, 0, ""

    # Load category tree
    try:
        with open('Data/category_tree.json', 'r') as f:
            category_tree = json.load(f)
    except FileNotFoundError:
        st.error("Error: `Data/category_tree.json` not found. Please ensure the file exists.")
        return None, 0, ""
    except json.JSONDecodeError:
        st.error("Error: `Data/category_tree.json` is not a valid JSON file.")
        return None, 0, ""
        
    # Overture Maps data fetching
    st.write("Fetching POI data from Overture Maps...")
    try:
        place_dataset = core.geodataframe("place", bbox=bbox_tuple)
        if place_dataset.empty:
            st.warning("No POIs found for the given BBOX.")
            return [], 0, "No POIs found."
        st.write(f"Found {len(place_dataset)} POIs.")
    except Exception as e:
        st.error(f"Error fetching data from Overture Maps: {e}")
        return None, 0, ""

    # Extract websites
    websites_to_scrape = {} # Store as {poi_primary_name: [website_url]}
    for i in range(len(place_dataset.id)):
        poi_name = place_dataset.names[i].get('primary', f"Unnamed POI {i}")
        websites_list = place_dataset.websites[i]
        websites_to_scrape[poi_name] = websites_list # websites_list can be None or list of URLs

    # Scrape websites
    st.write("Scraping websites...")
    g_scraped_sites_count = 0
    g_total_sites = len(websites_to_scrape)
    g_site_progress_bar.progress(0)
    g_site_progress_text.caption(f"Scraping websites: 0/{g_total_sites} (0.0%)")
    
    scraped_websites_data = {}
    # Prepare items for ThreadPoolExecutor: (poi_name, list_of_website_urls)
    website_items_for_executor = list(websites_to_scrape.items())

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor: # Reduced max_workers
        results = list(executor.map(scrape_site_wrapper_st, website_items_for_executor))
    scraped_websites_data = dict(results)
    g_site_progress_text.caption(f"Website scraping complete: {g_scraped_sites_count}/{g_total_sites}")


    # Extract and scrape social media links
    st.write("Scraping social media links...")
    socials_dict = {} # {poi_name: list_of_social_urls}
    for i in range(len(place_dataset.id)):
        place_name = place_dataset.names[i].get('primary', f"Unnamed POI {i}")
        social_links = place_dataset.socials[i] # This could be list, string, or None
        socials_dict[place_name] = social_links

    tasks_for_social_scraping = []
    for place_name, social_links_list in socials_dict.items():
        if social_links_list:
            if isinstance(social_links_list, (list, np.ndarray)):
                for social_url in social_links_list:
                    if isinstance(social_url, str) and social_url.startswith(('http://', 'https://')):
                        tasks_for_social_scraping.append((place_name, social_url))
            elif isinstance(social_links_list, str) and social_links_list.startswith(('http://', 'https://')):
                tasks_for_social_scraping.append((place_name, social_links_list))

    g_scraped_socials_count = 0
    g_total_social_urls = len(tasks_for_social_scraping)
    g_social_progress_bar.progress(0)
    g_social_progress_text.caption(f"Scraping social URLs: 0/{g_total_social_urls} (0.0%)")

    scraped_socials_content = {} # {poi_name: [{url: content}, ...]}
    if g_total_social_urls > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor: # Reduced max_workers
            social_results = list(executor.map(scrape_social_url_wrapper_st, tasks_for_social_scraping))
        for place_name, url, content in social_results:
            if place_name not in scraped_socials_content:
                scraped_socials_content[place_name] = []
            scraped_socials_content[place_name].append({url: content})
    g_social_progress_text.caption(f"Social media scraping complete: {g_scraped_socials_count}/{g_total_social_urls}")


    # Rule-based categorization
    poi_rule_categories = {}
    for poi_name, site_text in scraped_websites_data.items():
        social_entries = scraped_socials_content.get(poi_name, [])
        combined_social_text = " ".join(next(iter(d.values()), "") for d in social_entries if d)
        poi_rule_categories[poi_name] = rule_based_category(poi_name, site_text, combined_social_text)

    # Prepare socials string for LLM context (as in original script)
    formatted_social_info_list = []
    if isinstance(scraped_socials_content, dict):
        for place_name_key, list_of_content_dicts in scraped_socials_content.items():
            all_scraped_texts = [text for d in list_of_content_dicts for text in d.values() if text and isinstance(text, str)]
            aggregated_social_info = " | ".join(all_scraped_texts)
            place_name_escaped = str(place_name_key).replace('"', '\\"')
            aggregated_social_info_escaped = aggregated_social_info.replace('"', '\\"')
            formatted_social_info_list.append(f'"{place_name_escaped}": "{aggregated_social_info_escaped}"')
    scraped_socials_string_for_llm = ", ".join(formatted_social_info_list)


    # First LLM call: Category Prediction
    st.write("Requesting category predictions from LLM...")
    context1 = (
        "You are a Category Suggestion Agent. Your job is to read a POI record that I provide you with that consists of the POI name, "
        "rule-based search results and data scraped from its website and socials. Now using this propose a category for it which is as "
        "detailed as possible (Outputting 'italian restaurant' instead of 'restaurant') using its name, rule based search and web/socials "
        "scraped information, just categorize until you are sure about the category."
        f"These are all the categories in the schema in a JSON file that has all the categories in the form as a tree: {json.dumps(category_tree)}."
        "While categorizing, I expect you to traverse the category tree and categorize level by level. For example, if you are categorizing a restaurant, "
        "you should first categorize it as 'eat_and_drink', then as 'restaurant', and finally as 'italian restaurant' if applicable."
        "I am going to provide you with all the POIs names with the website scraped information, in this format: 'name': 'website scraped information'. "
        f"POIs with website scraped information: {json.dumps(scraped_websites_data)}"
        "I am also going to provide you with all the POIs names with the rule-based search results, in this format: 'name': 'rule-based search category prediction'. "
        f"POIs with rule-based search results: {json.dumps(poi_rule_categories)}"
        "Finally, I am going to provide you with all the POIs names with the socials scraped information, in this format: 'name': 'socials scraped information'. "
        f"POIs with socials scraped information: {scraped_socials_string_for_llm}"
        "Provide me with a category that best suits the primary category for these POIs from the list of categories I provided you with. "
        "Choose as detailed as possible and ONLY choose the categories that you are mostly sure about. "
        "The response should be in this format: 'name': 'predicted category & confidence percentage'. Be as honest as possible and if you are not sure "
        "about the category please say so. Do not provide any other information or explanation, just the POI's name and their predicted categories. "
        "Provide one line for each POI."
    )
    
    try:
        response1 = model.generate_content(context1)
        raw_llm_predictions_text = response1.text
    except Exception as e:
        st.error(f"Error during first LLM call: {e}")
        return None, 0, ""

    # Parse LLM response
    predicted_poi_data_for_json = []
    llm_predicted_categories_only = [] # For the second LLM call

    if raw_llm_predictions_text:
        for line in raw_llm_predictions_text.split('\n'):
            line = line.strip()
            if ':' in line:
                try:
                    parts = line.split(':', 1)
                    poi_name = parts[0].strip().strip("'\"")
                    prediction_full = parts[1].strip().strip("'\"") # "category & confidence%"
                    
                    predicted_poi_data_for_json.append({'poi_name': poi_name, 'prediction': prediction_full})
                    
                    # Extract just the category for the accuracy LLM
                    category_part = prediction_full.split(' & ')[0].strip().strip("'\"")
                    llm_predicted_categories_only.append(category_part)
                except Exception as e:
                    st.warning(f"Could not parse LLM prediction line: {line} - Error: {e}")
            elif line: # Fallback for lines that might just be a category (less ideal)
                category_part = line.strip().strip("'\"")
                llm_predicted_categories_only.append(category_part)
                # For JSON, we might not have a POI name here, so we could add with "Unknown POI" or skip
                # predicted_poi_data_for_json.append({'poi_name': 'Unknown POI (from simple line)', 'prediction': category_part})


    if not predicted_poi_data_for_json:
        st.warning("LLM did not return any parseable predictions.")
        # return [], 0, raw_llm_predictions_text # Return empty list if no predictions

    # Extract primary categories from Overture data for comparison
    primary_categories_list = []
    for index, poi_row in place_dataset.iterrows():
        primary_category = "N/A"
        if 'categories' in poi_row and poi_row['categories'] and isinstance(poi_row['categories'], dict) and 'primary' in poi_row['categories']:
            primary_category = poi_row['categories']['primary']
        primary_categories_list.append(primary_category)
    
    primary_categories_string_for_llm = "\n".join(primary_categories_list)
    predicted_categories_string_for_llm = "\n".join(llm_predicted_categories_only)

    # Second LLM call: Accuracy Assessment
    st.write("Requesting accuracy assessment from LLM...")
    accuracy_percentage = 0.0
    if predicted_categories_string_for_llm and primary_categories_string_for_llm :
        context2 = (
            "You are a category comparison agent. I will provide you with a list of categories that a software predicted for POIs "
            "and a list of the actual categories of those POIs. Compare each pair of predicted and actual categories and determine if they are similar. "
            "For example, 'active_life' and 'sports_and_recreation_venue' are similar. "
            f"Software predicted categories (one per line):\n{predicted_categories_string_for_llm}\n"
            f"Actual POI categories (one per line):\n{primary_categories_string_for_llm}\n"
            "For each pair, respond with 'yes' if they are similar, or 'no' if they are not. Provide only 'yes' or 'no' on each line, corresponding to each pair. "
            "Do not provide any other information or explanation."
        )
        try:
            response2 = model.generate_content(context2)
            comparison_results_text = response2.text
            
            answers = [ans.strip().lower() for ans in comparison_results_text.strip().split('\n') if ans.strip()]
            if answers:
                yes_answers = answers.count('yes')
                total_answers = len(answers)
                if total_answers > 0:
                    accuracy_percentage = (yes_answers / total_answers) * 100
            else:
                st.warning("LLM provided no comparison results for accuracy calculation.")

        except Exception as e:
            st.error(f"Error during second LLM call (accuracy): {e}")
    else:
        st.warning("Not enough data to perform LLM accuracy assessment (predicted or primary categories missing).")

    return predicted_poi_data_for_json, accuracy_percentage, raw_llm_predictions_text


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="POI Categorization", layout="wide")
    st.title("POI Categorization Engine")

    st.sidebar.header("Inputs")
    api_key_input = st.sidebar.text_input("Google Generative AI API Key:", type="password", help="Your API key for Google's Gemini models.")

    st.sidebar.subheader("Bounding Box (BBOX)")
    # Default values from the original script
    min_lon = st.sidebar.number_input("Min Longitude", format="%.4f", value=9.0894)
    min_lat = st.sidebar.number_input("Min Latitude", format="%.4f", value=45.5042)
    max_lon = st.sidebar.number_input("Max Longitude", format="%.4f", value=9.1094)
    max_lat = st.sidebar.number_input("Max Latitude", format="%.4f", value=45.5172)

    bbox_input = (min_lon, min_lat, max_lon, max_lat)

    if st.sidebar.button("🚀 Process POIs", use_container_width=True):
        if not api_key_input:
            st.error("🚨 Please enter your Google Generative AI API Key.")
        elif not all(isinstance(coord, float) for coord in bbox_input) or len(bbox_input) != 4:
            st.error("🚨 Please enter valid BBOX coordinates.")
        else:
            st.info(f"Processing POIs for BBOX: {bbox_input}...")
            
            # Placeholders for progress bars
            st.subheader("📊 Scraping Progress")
            col1, col2 = st.columns(2)
            with col1:
                st_site_progress_text = st.empty()
                st_site_progress_bar = st.progress(0)
            with col2:
                st_social_progress_text = st.empty()
                st_social_progress_bar = st.progress(0)

            streamlit_progress_elements = {
                "websites": (st_site_progress_text, st_site_progress_bar),
                "socials": (st_social_progress_text, st_social_progress_bar)
            }
            
            with st.spinner("🧠 Performing analysis and LLM magic... This can take several minutes."):
                predicted_data, accuracy, raw_llm_output = process_poi_data(
                    bbox_input, 
                    api_key_input,
                    streamlit_progress_elements
                )

            if predicted_data is not None: # Check if processing was successful (not None)
                st.success("✅ Processing Complete!")
                
                st.subheader("🎯 Predicted Categories & Confidence")
                if predicted_data:
                    df_predictions = pandas.DataFrame(predicted_data)
                    st.dataframe(df_predictions, use_container_width=True)

                    json_data_to_download = json.dumps(predicted_data, indent=2)
                    st.download_button(
                        label="📥 Download Predictions (JSON)",
                        data=json_data_to_download,
                        file_name=f"poi_predictions_bbox_{min_lon}_{min_lat}_{max_lon}_{max_lat}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No predictions were generated by the LLM, or no POIs found.")

                st.subheader("📈 LLM-Assessed Similarity Accuracy")
                st.metric(label="Accuracy", value=f"{accuracy:.2f}%")
                st.caption("This accuracy is based on the LLM's assessment of similarity between its own predictions and Overture's primary categories.")

                with st.expander("Raw LLM Output (Category Predictions)"):
                    st.text_area("LLM Output:", raw_llm_output, height=200, 
                                 help="This is the raw text output from the first LLM call for category predictions.")
            else:
                st.error("Processing failed. Please check the console for error messages if running locally, or review input parameters.")
    else:
        st.info("Enter BBOX coordinates and your API key, then click 'Process POIs' to begin.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with [Streamlit](https://streamlit.io) and [Overture Maps](https://overturemaps.org).")


if __name__ == "__main__":
    main()
