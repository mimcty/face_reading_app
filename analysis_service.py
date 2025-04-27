from openai import OpenAI
from typing import Dict, Optional, Union, List
import os
import re
from datetime import datetime
from .face_detector import FaceDetector
from .personality_matcher import PersonalityMatcher
from .image_generator import ImageGenerator


class FaceReadingAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.ai_client = OpenAI(api_key=api_key)
        self.detector = FaceDetector()
        self.matcher = PersonalityMatcher()
        self.image_gen = ImageGenerator(api_key)

        # Classical texts database
        self.classical_texts = {
            "forehead": [
                "《麻衣神相》卷三：'额如覆肝者贵，如立壁者富'",
                "《神相全编》：'天庭饱满，少年得志'"
            ],
            "eyes": [
                "《柳庄相法》：'凤目细长，必做王侯'",
                "《冰鉴》：'目为心之窗，睛定则心定'"
            ],
            # ... other feature references
        }

    def full_analysis(self, image_path: Union[str, bytes], preferences: Dict) -> Dict:
        """Complete premium face reading analysis"""
        try:
            # Step 1: Face detection and measurement
            image_data = self._load_image_data(image_path)
            detection_result = self.detector.get_measurements(image_data)

            if not detection_result or "error" in detection_result:
                return {"error": "面相测量失败，请提供清晰正面照片"}

            # Step 2: Prepare analysis data
            element = detection_result.get("element", "unknown")
            shape = detection_result.get("shape", "unknown")

            measurements = self._prepare_measurements(detection_result, element, shape)

            # Step 3: Get in-depth AI analysis
            ai_analysis = self._get_master_analysis(
                measurements=measurements,
                preferences=preferences,
                dominant_element=element
            )

            # Step 4: Generate ideal partner image + explanation
            try:
                # Combine all data needed for generation
                generation_input = {
                    **measurements,
                    "preferences": preferences,
                    "personality_type": preferences.get("personality", "未指定")
                }

                partner_image_url, partner_analysis = self.image_gen.generate_partner_with_analysis(generation_input)
                partner_data = {
                    "image_url": partner_image_url,
                    "analysis": partner_analysis
                }
            except Exception as e:
                partner_data = {}
                print("[WARN] Partner generation failed:", str(e))

            # Step 5: Compile final report
            report = self._compile_full_report(
                measurements,
                ai_analysis,
                preferences
            )
            report["ideal_partner"] = partner_data

            print("[DEBUG] Analysis content: ", report)
            return report

        except Exception as e:
            return {"error": f"至尊面相分析失败: {str(e)}"}
    def _prepare_measurements(self, detection_result: Dict, element: str, shape: str) -> Dict:
        """Prepare measurement data with philosophical insights"""
        return {
            "scientific_analysis": detection_result.get("measurements", {}),
            "traits": detection_result.get("traits", {}),
            "chinese_reading": {
                "element": element,
                "shape": shape,
                "confidence": detection_result.get("confidence", {}),
                "breakdown": self._enrich_breakdown(detection_result.get("element_breakdown", {})),
                "scores": detection_result.get("element_scores", {})
            },
            "additional_metrics": {
                "facial_symmetry": self._calculate_harmony_score(detection_result),
                "qi_circulation": self._assess_qi_flow(detection_result)
            }
        }

    def _enrich_breakdown(self, breakdown: Dict) -> Dict:
        """Add philosophical insights to each feature"""
        enriched = {}
        feature_insights = {
            "face_shape": "面形乃先天之气与后天修养共同塑造",
            "eyes": "目为心之窗，藏神之所",
            "eyebrows": "眉为保寿官，主兄弟情谊",
            "nose": "鼻为审辨官，关乎中年运势",
            "mouth": "口为出纳官，主饮食福禄",
            "jawline": "地阁方圆，晚运亨通"
        }

        for feature, (elem, reason) in breakdown.items():
            insight = feature_insights.get(feature, "相理微妙，需结合整体观之")
            enriched[feature] = (elem, f"{reason}（{insight}）")

        return enriched

    def _get_master_analysis(self, measurements: Dict, preferences: Dict, dominant_element: str) -> Dict:
        """Get ultra-detailed analysis from AI master"""
        prompt = self._create_master_prompt(measurements, preferences, dominant_element)

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": self._create_master_instructions()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2500,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            return self._parse_master_response(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

    def _create_master_instructions(self) -> str:
        """System instructions for the AI master"""
        return """你是一位拥有60年经验的面相宗师，麻衣神相第39代传人。你的分析需要：
        1. 严格遵循古典相法原则
        2. 每项判断必须引用典籍
        3. 预测需具体到年龄段
        4. 提供可验证的建议
        5. 语言庄严而富有智慧
        
        使用以下分析框架：
        【三停五岳总论】
        - 上停：早年运势（0-28岁）
        - 中停：中年运程（28-50岁） 
        - 下停：晚年福禄（50岁后）
        - 五岳：人生各领域平衡度
        
        【五官精微分析】
        1. 眉（保寿宫）：性情/兄弟缘
        2. 眼（监察官）：心智/贵人运  
        3. 鼻（审辨官）：财运/健康
        4. 口（出纳官）：食禄/人际关系
        5. 耳（采听官）：福气/寿元
        
        【十二宫详批】
        逐宫分析人生各领域...
        
        【五行命理精解】
        1. 性格深层分析
        2. 职业方向建议
        3. 婚配宜忌
        4. 健康预警
        5. 大运走势
        
        【改运秘法】
        具体可行的改善建议"""

    def _create_master_prompt(self, measurements: Dict, preferences: Dict, dominant_element: str) -> str:
        """Create the detailed prompt for master analysis"""
        current_year = datetime.now().year
        gender = preferences.get("partner_gender", "unknown")
        gender_label = "男性" if gender == "male" else "女性"
        age_range = preferences.get("age_range", [25, 45])
        age_range_str = f"{age_range[0]}至{age_range[1]}岁"
        formatted_features = self._format_measurements_for_prompt(measurements)

        return f"""
            【至尊面相分析请求】

            弟子今日为一位{gender_label}求测者进行面相分析，当前为{current_year}年，流年主{dominant_element}行大运。
            求测者年龄范围为 {age_range_str}，请结合其阶段重点分析其运势，给予段落式详细说明，而非简略描述或单句判断。

            ======== 求测者基础面相信息 ========
            • 主五行：{measurements['chinese_reading']['element']}
            • 面形格局：{measurements['chinese_reading']['shape']}
            • 五行得分概览：{measurements['chinese_reading']['scores']}

            ======== 面相特征分析依据 ========
            {formatted_features}

            ======== 分析要求 ========
            请严格按照以下要求进行至尊级分析：
            1. 每项特征需引用至少两处古籍经典（如《麻衣神相》《柳庄相法》《冰鉴》等）
            2. 每段落需不少于三句，每个面相区域解释应形成完整、正式段落
            3. 对于上中下三停运势变化应分别论述，并结合求测者年龄阶段重点展开
            4. 提供基于五行与面相综合的具体改运建议，需通俗但不失权威
            5. 字数建议超过2500字

            请以庄严古法语气，撰写一份可用于专业咨询服务的命理分析报告：
            """

    def _format_measurements_for_prompt(self, measurements: Dict) -> str:
        """Format measurements for the AI prompt"""
        lines = []
        for feature, (elem, desc) in measurements["chinese_reading"]["breakdown"].items():
            lines.append(f"- {feature}: {desc} (属{elem})")

        return "\n".join(lines)

    def _parse_master_response(self, text: str) -> Dict:
        """Parse the AI master's response into structured data"""
        sections = {
            "three_divisions": r"【三停五岳总论】([\s\S]+?)(?=\n\n【|$)",
            "facial_features": r"【五官精微分析】([\s\S]+?)(?=\n\n【|$)",
            "twelve_palaces": r"【十二宫详批】([\s\S]+?)(?=\n\n【|$)",
            "element_analysis": r"【五行命理精解】([\s\S]+?)(?=\n\n【|$)",
            "enhancement": r"【改运秘法】([\s\S]+)"
        }

        result = {"professional_analysis": {}}

        # Extract main sections
        for section, pattern in sections.items():
            match = re.search(pattern, text)
            if match:
                result["professional_analysis"][section] = match.group(1).strip()

        # Extract key predictions
        result["key_predictions"] = self._extract_predictions(text)
        result["life_stages"] = self._analyze_life_stages(text)

        return result

    def _extract_predictions(self, text: str) -> List[Dict]:
        """Extract specific predictions from analysis text"""
        predictions = []

        # Extract age-specific predictions
        age_matches = re.finditer(r"(\d+)[\-至](\d+)岁[：:]([^\n]+)", text)
        for match in age_matches:
            predictions.append({
                "age_range": f"{match.group(1)}-{match.group(2)}",
                "prediction": match.group(3).strip(),
                "type": "age_based"
            })

        # Extract year-specific predictions
        year_matches = re.finditer(r"(\d{4})年[：:]([^\n]+)", text)
        for match in year_matches:
            predictions.append({
                "year": match.group(1),
                "prediction": match.group(2).strip(),
                "type": "year_specific"
            })

        return predictions

    def _analyze_life_stages(self, text: str) -> Dict:
        """Analyze and structure life stage information"""
        stages = {
            "early_life": {"range": "0-28", "content": ""},
            "mid_life": {"range": "28-50", "content": ""},
            "late_life": {"range": "50+", "content": ""}
        }

        # Extract early life (上停)
        early_match = re.search(r"上停[^\n]+[\s\S]+?(?=中停|$)", text)
        if early_match:
            stages["early_life"]["content"] = early_match.group(0).strip()

        # Extract mid life (中停)
        mid_match = re.search(r"中停[^\n]+[\s\S]+?(?=下停|$)", text)
        if mid_match:
            stages["mid_life"]["content"] = mid_match.group(0).strip()

        # Extract late life (下停)
        late_match = re.search(r"下停[^\n]+[\s\S]+?(?=\n\n|$)", text)
        if late_match:
            stages["late_life"]["content"] = late_match.group(0).strip()

        return stages

    def _compile_full_report(self, measurements: Dict, analysis: Dict, preferences: Dict) -> Dict:
        """Compile all data into final report"""
        return {
            "metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "client_gender": preferences.get("gender", "unknown"),
                "partner_gender": preferences.get("partner_gender", "unknown"),
                "analysis_version": "premium_1.0"
            },
            "client_info": {
                "gender": preferences.get("gender", "unknown"),
                "partner_gender": preferences.get("partner_gender", "unknown"),
                "age_range": preferences.get("age_range", [25, 45])
            },
            "measurements": measurements,
            "professional_analysis": analysis["professional_analysis"],
            "predictions": {
                "key_years": [p for p in analysis["key_predictions"] if p["type"] == "year_specific"],
                "age_ranges": [p for p in analysis["key_predictions"] if p["type"] == "age_based"],
                "life_stages": analysis["life_stages"]
            },
            "recommendations": self._extract_recommendations(analysis["professional_analysis"].get("enhancement", ""))
        }

    def _extract_recommendations(self, enhancement_text: str) -> Dict:
        """Extract structured recommendations from enhancement section"""
        categories = {
            "career": r"事业建议[：:]([^\n]+)",
            "health": r"养生建议[：:]([^\n]+)",
            "relationships": r"人际建议[：:]([^\n]+)",
            "wealth": r"招财方法[：:]([^\n]+)"
        }

        recommendations = {}
        for category, pattern in categories.items():
            match = re.search(pattern, enhancement_text)
            if match:
                recommendations[category] = match.group(1).strip()

        return recommendations

    # Helper methods
    def _calculate_harmony_score(self, detection_result: Dict) -> float:
        """Calculate facial harmony score (0-100)"""
        # Implementation depends on your face detector metrics
        return min(100, max(0,
                            detection_result.get("symmetry", {}).get("overall_symmetry", 0.5) * 100
                            ))

    def _assess_qi_flow(self, detection_result: Dict) -> str:
        """Assess Qi circulation based on facial features"""
        complexion = detection_result.get("qi", {}).get("complexion", "normal")
        luster = detection_result.get("qi", {}).get("luster_score", 0)

        if luster > 30 and complexion == "reddish":
            return "气血旺盛，需注意疏导"
        elif luster < 20:
            return "气血不足，建议调理"
        else:
            return "气血调和，状态良好"

    def _load_image_data(self, image_path: Union[str, bytes]) -> bytes:
        """Load image data from path or bytes"""
        if isinstance(image_path, str):
            with open(image_path, 'rb') as f:
                return f.read()
        return image_path