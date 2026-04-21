import json

input_path = "sharegpt4video_subset.json"
output_path = "sharegpt4video_subset_distilled.json"

metadata_map = {
    "01ff5bb49c2ff92fa41f98dbd1da45a4e9653d215161de8f58647e87af7daddd": {
        "tags": ["tranquil lake", "evergreen forest", "rolling hills", "rugged mountains", "water ripples", "sunlight glisten", "trees", "sky", "clouds", "nature"],
        "mood": ["serene", "peaceful", "undisturbed", "calm", "natural", "scenic"],
        "action": ["rippling water", "glistening sunlight", "static camera", "nature observation"]
    },
    "056ca2389ab78e72845916ddd54baba871f26c009591c3715e1abb9fa59b526f": {
        "tags": ["black screen", "particle lights", "star-like spots", "cartoon character", "green creature", "magenta hat", "blue silhouette", "animation", "abstract art"],
        "mood": ["mysterious", "whimsical", "abstract", "thoughtful", "puzzled", "playful"],
        "action": ["fading in", "appearing", "disappearing", "transitioning", "illuminating"]
    },
    "05938294c6c0f3be81ee3b76d3596ee621df5839017294f9f9ff9bbed3146dc3": {
        "tags": ["smoke cloud", "human silhouette", "white background", "abstract figure", "mist", "grayscale tones"],
        "mood": ["mysterious", "ethereal", "abstract", "transformative", "artistic", "ambiguous"],
        "action": ["morphing", "billowing", "dissipating", "transforming", "swirling", "blending"]
    },
    "0b5dd3ea2ca9a228ca4b333d7cb4b47995f9313082928cba28161747dcd5b5c4": {
        "tags": ["deep blue ocean", "undulating waves", "white sea foam", "aerial view", "water surface", "sea crests"],
        "mood": ["turbulent", "dynamic", "natural", "powerful", "restless", "hypnotic"],
        "action": ["flowing", "crashing", "undulating", "churning", "foaming", "moving"]
    },
    "5a00cd8770fc618aa630ff3aecd868b9258404c4727b4c8b20c42c814aa6b08f": {
        "tags": ["cobblestone street", "pedestrians", "feet walking", "various shoes", "bicycles", "storefront", "sunlight shadows", "urban setting", "public space"],
        "mood": ["urban", "observational", "lively", "peaceful", "busy", "everyday life"],
        "action": ["walking", "stepping", "passing", "crossing", "shadows moving", "people commuting"]
    },
    "faddc9ff55285d0ff5a14429ee4afc5a8c902363179bbb89acb36854d5f20897": {
        "tags": ["ancient ruins", "stone columns", "weathered stone", "archaeological site", "greenery", "rubble", "historic architecture", "sunlight"],
        "mood": ["ancient", "still", "historic", "timeless", "serene", "majestic"],
        "action": ["standing still", "casting shadows", "nature reclaiming", "static observation"]
    },
    "1c7a69eb535bfc001f695c11ed202542c0a8d0432912c7212459c7c1bc4741ee": {
        "tags": ["CPR demonstration", "medical mannequin", "healthcare professional", "chest compressions", "medical training", "exam table", "stethoscope", "hands"],
        "mood": ["educational", "clinical", "serious", "focused", "professional", "instructional"],
        "action": ["compressing chest", "checking airway", "demonstrating cpr", "training", "practicing"]
    },
    "159179ef78b73848ce99fe3d0bb2c8f69329789be146e2f17da7183dc4ba753e": {
        "tags": ["flowing river", "tree-lined banks", "ornate railing", "water reflections", "sunlight shimmering", "green foliage", "channel", "nature"],
        "mood": ["calm", "scenic", "relaxing", "tranquil", "peaceful", "harmonious"],
        "action": ["flowing", "shimmering", "reflecting light", "glistening", "static view"]
    },
    "276a1b553450551204df8fabbae1eacc7409b221e0c2548d21cf469ee1f5d131": {
        "tags": ["mystical forest", "fairy figure", "green skin", "blue hair", "flowing white dress", "glowing hand", "butterfly", "magical sparkles", "mist", "fantasy art"],
        "mood": ["fantasy", "magical", "enchanted", "ethereal", "whimsical", "surreal", "mystical"],
        "action": ["holding butterfly", "casting spell", "glowing", "summoning magic", "standing still", "emitting light"]
    },
    "00f855be8812dbf6fb3d62bc32c8ef356dde9c410a9da15de02a01f667903be4": {
        "tags": ["shipping port", "cargo containers", "cargo ship", "large crane", "dock operations", "trucks", "forklifts", "aerial view", "industrial site", "water"],
        "mood": ["industrial", "busy", "organized", "logistical", "bustling", "systematic"],
        "action": ["camera panning", "zooming", "loading cargo", "vehicles moving", "crane operating", "shipping"]
    }
}

try:
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Process data
    for item in data:
        video_id = item.get("video_id")
        
        # Find reasoning
        reasoning = ""
        if "captions" in item:
            for cap in item["captions"]:
                if cap.get("idx") == "-1":
                    reasoning = cap.get("content", "")
                    break
        
        item["reasoning"] = reasoning
        
        # Add index
        if video_id in metadata_map:
            item["index"] = metadata_map[video_id]
        else:
            # Default empty if ID not found (shouldn't happen for this subset)
            item["index"] = {"tags": [], "mood": [], "action": []}
            
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully processed {len(data)} items into {output_path}")

except Exception as e:
    print(f"Error: {e}")
