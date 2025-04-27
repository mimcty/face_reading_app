from openai import OpenAI
import os
import requests
from typing import Dict, Tuple
from pathlib import Path
import random


class ImageGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path("generated")
        self.output_dir.mkdir(exist_ok=True)

        # Chinese face reading compatibility rules
        self.element_compatibility = {
            "metal": {
                "complement": "water",
                "features": ["round face", "soft cheeks", "delicate nose"],
                "strengths": ["social grace", "organization"],
                "balance": ["more spontaneity", "flexibility"]
            },
            "wood": {
                "complement": "fire",
                "features": ["heart-shaped face", "expressive eyebrows"],
                "strengths": ["creativity", "vision"],
                "balance": ["more grounding", "patience"]
            },
            "water": {
                "complement": "earth",
                "features": ["full cheeks", "rounded jawline"],
                "strengths": ["wisdom", "adaptability"],
                "balance": ["more assertiveness", "decisiveness"]
            },
            "fire": {
                "complement": "metal",
                "features": ["pointed chin", "angular features"],
                "strengths": ["passion", "leadership"],
                "balance": ["more calmness", "reflection"]
            },
            "earth": {
                "complement": "wood",
                "features": ["square jaw", "broad forehead"],
                "strengths": ["stability", "reliability"],
                "balance": ["more creativity", "spontaneity"]
            }
        }

    def generate_partner_with_analysis(self, user_analysis: Dict) -> Tuple[str, str]:
        """
        Generate partner image and detailed analysis
        Returns:
            tuple: (image_path, analysis_description)
        """
        print("Image Generator Received:", user_analysis.keys())
        if 'chinese_reading' in user_analysis:
            print("Chinese Reading Contains:", user_analysis['chinese_reading'].keys())

        # Generate the complementary image
        image_url = self._generate_complementary_image(user_analysis)
        image_path = self._save_image(image_url)

        # Generate Master Chen's analysis
        analysis_text = self._generate_analysis(
            user_analysis=user_analysis,
            partner_features=self._ai_generate_partner_traits(user_analysis)
        )

        return image_path, analysis_text

    def _generate_complementary_image(self, analysis: Dict) -> str:
        """Generate the complementary partner image"""
        prompt = self._build_image_prompt(analysis)
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            style="natural"
        )
        return response.data[0].url

    def _generate_analysis(self, user_analysis: Dict, partner_features: Dict) -> str:
        """Generate Master Chen's face reading analysis"""
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": """You are a 60-year-old master of Chinese physiognomy (面相学) from Beijing. 
                    Structure your response with headers so that it is easy for the reader to follow.
                    Your analysis should follow these traditional structures:

                    面相分析结构:
                    1. 五行互补 (Elemental Balance)
                    - Explain the elemental interaction using Wu Xing theory
                    - Describe how the partner's element complements the user's
                    - Explain why the partner is a good fit for the user

                    2. 面相特征 (Facial Features Analysis)
                    - Explain 3 key facial features that the user's partner will have and why.
                    - Analyze without using numbers. Pretend you are analyzing like a real Chinese master, in person.
                    - Relate each to personality and destiny according to classical texts
                    - Mention both positive aspects and cautions

                    3. 夫妻宫匹配 (Marriage Palace Compatibility)
                    - Assess compatibility between the user and their ideal partner based on forehead, eyebrows, and nose bridge
                    - Reference the 12 Palaces of 面相

                    4. 运势建议 (Fortune Guidance)
                    - Provide specific advice for the relationship
                    - Explain the future of the user's relationship with their partner
                    - What are some challenges that they can face in the relationship?
                    - What are the strengths of their relationship?
                    - How to overcome challenges in the relationship
                    - Suggest auspicious directions/colors

                    Style Requirements:
                    - 800-1000 words
                    - Include 1-2 classical references 
                    - Sound like a street fortune teller (warm, personal, slightly mysterious)
                    - Never use modern psychological terms"""
                },
                {
                    "role": "user",
                    "content": f"""
                    [求测者信息]
                    五行: {user_analysis['chinese_reading']['element']}
                    面相特征: {user_analysis['scientific_analysis']}
                    性格: {user_analysis.get('personality_type', '未指定')}

                    [配对对象]
                    互补五行: {partner_features['complementary_element']}
                    推荐特征: {partner_features['features']}

                    Generate a reading that would be given at a traditional face reading stall in Beijing's Temple of Heaven park.
                    """
                }
            ],
            temperature=0.6,  # Slightly lower for more consistency
            max_tokens=1800
        )
        return response.choices[0].message.content

    def _build_image_prompt(self, analysis: Dict) -> str:
        """Build DALL-E prompt for complementary partner"""
        try:
            user_element = analysis.get('chinese_reading', {}).get('element', 'earth').lower()
            if user_element not in self.element_compatibility:
                user_element = 'earth'

            comp_data = self._ai_generate_partner_traits(analysis)
            traits = comp_data["features"]

            return f"""
            Create a highly photorealistic portrait of a {comp_data['gender']} partner based on Chinese face reading:

            [Purpose]
            - Design a {comp_data['gender']} partner who complements the user's {comp_data['complementary_element']}-element face, incorporating harmonious features to create balance.
            - Harmonize with user's {analysis.get('personality_type', '')} personality

            [Key Features]
            - Gender: Clearly {comp_data['gender']}
            - Race: Matches the user's racial background
            - Age: Partner must look around the user's age as seen in the uploaded photo
            - Eyes: {traits.get('eyes', 'gentle eyes')}
            - Nose: {traits.get('nose', 'balanced nose')}
            - Mouth: {traits.get('lips', 'soft lips')}
            - Face Shape: {traits.get('face shape', 'harmonious face')}
            - Eyebrows: {traits.get('eyebrows', 'natural brows')}
            - Energy: {traits.get('expression', 'calm and steady')}
            - Energy: Balanced, {random.choice(['calm', 'confident', 'serene'])} with {random.choice(['depth', 'mystery', 'tenderness'])}
            - Modern hairstyle appropriate for their age
            - Photographed from shoulder level up
            - Looking directly at camera
            - Wearing contemporary casual or business casual attire

            [Style]
            - Contemporary clothing and styling (2020s era)
            - Perfect focus with subtle depth of field
            - Natural lighting with soft shadows
            - Detailed skin texture without artificial smoothing
            - Modern professional portrait photography
            - Hyperrealistic style, indistinguishable from a real photograph
            - Professional camera quality (Canon/Sony/Nikon look)
            - No filters or artistic effects
            """
        except KeyError as e:
            raise ValueError(f"Missing required data in analysis: {str(e)}")

    def _ai_generate_partner_traits(self, analysis: Dict) -> Dict:
        """Use AI to dynamically generate ideal partner facial traits based on Chinese compatibility"""
        user_element = analysis.get("chinese_reading", {}).get("element", "earth")
        face_traits = analysis.get("traits", {})
        scientific = analysis.get("scientific_analysis", {})

        gender = analysis['preferences'].get('partner_gender', 'female').lower()
        gender = gender if gender in ['male', 'female'] else 'female'

        prompt = f"""
        You are a master of Chinese face reading (面相学). Based on the user's facial traits and dominant element,
        describe the ideal *visual traits* their romantic partner should have, including:
        - Eyes (shape, intensity)
        - Nose (bridge shape, size)
        - Lips (shape, softness)
        - Face shape (structure, jawline)
        - Eyebrows (angle, thickness)
        - Energy and expression (warm, serious, mysterious, bright, calm, etc.)

        Always respond with structured markdown like:
        ```
        - Eyes: ...
        - Nose: ...
        - Lips: ...
        - Face shape: ...
        - Eyebrows: ...
        - Expression: ...
        ```

        Focus on balancing and complementing the user's traits:
        - User’s dominant element: {user_element}
        - User's measured traits: {face_traits}
        - User’s geometry: {scientific}
        """

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system",
                 "content": "You are a 60-year-old Chinese face reader with 40 years of experience. Use traditional principles, not Western psychology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )

        result = response.choices[0].message.content.strip()

        # Basic parse to extract features
        traits = {}
        for line in result.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                traits[key.strip().lower()] = value.strip()

        return {
            "features": traits,
            "gender": gender,
            "complementary_element": user_element
        }

    def _save_image(self, image_url: str) -> str:
        """Download and save the image"""
        filename = f"partner_{os.urandom(4).hex()}.png"
        path = self.output_dir / filename
        with open(path, "wb") as f:
            f.write(requests.get(image_url).content)
        print("Generated image URL:", image_url)
        return f"http://localhost:8000/generated/{filename}"  # or your production domain