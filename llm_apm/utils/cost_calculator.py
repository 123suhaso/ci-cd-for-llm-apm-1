from typing import Optional, Dict, Any
import logging
import asyncio
from ..config.settings import get_config, register_model_pricing

logger = logging.getLogger(__name__)

class CostCalculator:
    """Enhanced utility class for calculating LLM API costs with auto-discovery"""

    def __init__(self):
        self.pricing_cache: Dict[str, Dict[str, float]] = {}
        self._config = get_config()
        
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int, 
                      request_monitor=None, context: Optional[Dict] = None) -> float:
     
        try:
            effective_model = self._resolve_model_name(model, request_monitor, context)
            
            pricing = self._config.get_model_pricing(effective_model)
            
            input_cost = (int(input_tokens) / 1000.0) * pricing["input"]
            output_cost = (int(output_tokens) / 1000.0) * pricing["output"]
            total_cost = input_cost + output_cost
            
            tier = self._config.estimate_model_tier(effective_model)
            logger.debug(
                f"Cost calculation for {effective_model} (tier={tier}): "
                f"Input={input_tokens} tokens (${input_cost:.8f}), "
                f"Output={output_tokens} tokens (${output_cost:.8f}), "
                f"Total=${total_cost:.8f}"
            )
            
            self.pricing_cache[effective_model] = pricing
            
            return round(total_cost, 8)
            
        except Exception as e:
            logger.error(f"Error calculating cost for model {model}: {e}")
            fallback_cost = self._emergency_cost_fallback(input_tokens, output_tokens)
            logger.warning(f"Using emergency fallback cost: ${fallback_cost:.6f}")
            return fallback_cost

    def _resolve_model_name(self, model: Optional[str], request_monitor=None, 
                           context: Optional[Dict] = None) -> str:
        """Resolve the actual model name from multiple sources"""
        
       
        
        if model and model.strip():
            return model.strip()
            
        if request_monitor and hasattr(request_monitor, 'metrics'):
            monitor_model = getattr(request_monitor.metrics, 'model', None)
            if monitor_model:
                return str(monitor_model).strip()
        
        if context and isinstance(context, dict):
            context_model = context.get('chosen_model') or context.get('model')
            if context_model:
                return str(context_model).strip()
                
        return self._config.default_model 

    def _emergency_cost_fallback(self, input_tokens: int, output_tokens: int) -> float:
        emergency_pricing = {"input": 0.005, "output": 0.015}  
        input_cost = (int(input_tokens) / 1000.0) * emergency_pricing["input"]
        output_cost = (int(output_tokens) / 1000.0) * emergency_pricing["output"]
        return round(input_cost + output_cost, 8)

    async def register_new_model(self, model: str, input_price: float, output_price: float,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new model with its pricing"""
        try:
            register_model_pricing(model, input_price, output_price, metadata)
            if model in self.pricing_cache:
                del self.pricing_cache[model]
            logger.info(f"Successfully registered new model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model}: {e}")
            return False

    async def discover_and_register_models(self, models_with_pricing: Dict[str, Dict[str, float]]) -> int:
        """Bulk register multiple models"""
        registered_count = 0
        for model, pricing in models_with_pricing.items():
            try:
                success = await self.register_new_model(
                    model, pricing["input"], pricing["output"]
                )
                if success:
                    registered_count += 1
            except Exception as e:
                logger.error(f"Failed to register {model}: {e}")
        
        logger.info(f"Registered {registered_count}/{len(models_with_pricing)} models")
        return registered_count

    

    def get_pricing_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive pricing info for a model"""
        try:
            pricing = self._config.get_model_pricing(model)
            tier = self._config.estimate_model_tier(model)
            
            return {
                "model": model,
                "pricing": pricing,
                "tier": tier,
                "input_price_per_1k": pricing["input"],
                "output_price_per_1k": pricing["output"],
                "source": "discovered" if model in self._config.get_discovered_models() else "static"
            }
        except Exception as e:
            logger.error(f"Error getting pricing info for {model}: {e}")
            return None

    def compare_model_costs(self, models: list, input_tokens: int, output_tokens: int) -> Dict[str, Dict[str, Any]]:
        """Compare costs across multiple models"""
        comparison = {}
        
        for model in models:
            try:
                cost = self.calculate_cost(model, input_tokens, output_tokens)
                pricing_info = self.get_pricing_info(model)
                
                comparison[model] = {
                    "total_cost": cost,
                    "tier": pricing_info["tier"] if pricing_info else "unknown",
                    "pricing": pricing_info["pricing"] if pricing_info else None
                }
            except Exception as e:
                logger.error(f"Error comparing cost for {model}: {e}")
                comparison[model] = {"error": str(e)}
        
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1].get("total_cost", float('inf')))
        )
        
        return sorted_comparison

    def get_cheapest_model_for_tier(self, tier: str) -> Optional[str]:
        """Find the cheapest model in a given tier"""
        try:
            models_in_tier = self._config.get_models_by_tier(tier)
            if not models_in_tier:
                return None
            
            costs = []
            for model in models_in_tier:
                cost = self.calculate_cost(model, 1000, 500)
                costs.append((model, cost))
            
            costs.sort(key=lambda x: x[1])
            return costs[0][0] if costs else None
            
        except Exception as e:
            logger.error(f"Error finding cheapest model for tier {tier}: {e}")
            return None

    def suggest_model_for_budget(self, max_cost_per_request: float, 
                               avg_input_tokens: int, avg_output_tokens: int) -> Optional[str]:
        """Suggest a model that fits within the budget"""
        try:
            all_models = self._config.get_all_known_models()
            suitable_models = []
            
            for model in all_models:
                cost = self.calculate_cost(model, avg_input_tokens, avg_output_tokens)
                if cost <= max_cost_per_request:
                    tier = self._config.estimate_model_tier(model)
                    suitable_models.append((model, cost, tier))
            
            if not suitable_models:
                return None
            
            tier_priority = {"enterprise": 0, "premium": 1, "standard": 2, "budget": 3}
            suitable_models.sort(key=lambda x: (tier_priority.get(x[2], 4), x[1]))
            
            recommended = suitable_models[0]
            logger.info(f"Recommended model for budget ${max_cost_per_request:.4f}: {recommended[0]} (${recommended[1]:.4f}, {recommended[2]} tier)")
            return recommended[0]
            
        except Exception as e:
            logger.error(f"Error suggesting model for budget: {e}")
            return None

    def refresh_config(self):
        """Refresh the config reference (useful after config updates)"""
        self._config = get_config()
        self.pricing_cache.clear()
        logger.info("Cost calculator config refreshed")

cost_calculator = CostCalculator()

async def register_model(model: str, input_price: float, output_price: float, 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
    return await cost_calculator.register_new_model(model, input_price, output_price, metadata)

def get_model_suggestions(max_budget: float, input_tokens: int = 1000, output_tokens: int = 500) -> Dict[str, Any]:
    """Get model suggestions within budget"""
    suggestion = cost_calculator.suggest_model_for_budget(max_budget, input_tokens, output_tokens)
    all_models = cost_calculator._config.get_all_known_models()
    comparison = cost_calculator.compare_model_costs(all_models[:5], input_tokens, output_tokens)  # Limit to first 5 for performance
    
    return {
        "recommended": suggestion,
        "budget": max_budget,
        "sample_comparison": comparison
    }