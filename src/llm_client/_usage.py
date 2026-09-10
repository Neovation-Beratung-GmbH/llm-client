from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from decimal import Decimal

from ._config import *

__all__ = ["UsageBreakdown", "CostBreakdown", "Usage", "get_usage"]

@dataclass(frozen=False)
class UsageBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    image_input_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: UsageBreakdown) -> UsageBreakdown:
        return UsageBreakdown(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def __str__(self) -> str:

        lines = ["Usage breakdown:"]
        if self.input_tokens:
            lines.append(f"  Input Tokens:          {self.input_tokens}")
        if self.output_tokens:
            lines.append(f"  Output Tokens:         {self.output_tokens}")
        if self.cached_tokens:
            lines.append(f"  Cached Tokens:         {self.cached_tokens}")
        if self.cache_creation_input_tokens:
            lines.append(f"  Cache Creation Tokens: {self.cache_creation_input_tokens}")
        if self.cache_read_input_tokens:
            lines.append(f"  Cache Read Tokens:     {self.cache_read_input_tokens}")
        if self.reasoning_tokens:
            lines.append(f"  Reasoning Tokens:      {self.reasoning_tokens}")
        if self.audio_input_tokens:
            lines.append(f"  Audio Input Tokens:    {self.audio_input_tokens}")
        if self.audio_output_tokens:
            lines.append(f"  Audio Output Tokens:   {self.audio_output_tokens}")
        if self.image_input_tokens:
            lines.append(f"  Image Input Tokens:    {self.image_input_tokens}")
        if self.total_tokens:
            lines.append(f"  Total Tokens:          {self.total_tokens}")
        return "\n".join(lines)
    

@dataclass(frozen=False)
class CostBreakdown:
    input_cost_USD: Decimal = Decimal(0)
    output_cost_USD: Decimal = Decimal(0)
    cached_input_cost_USD: Decimal = Decimal(0)
    cache_creation_cost_USD: Decimal = Decimal(0)
    cache_read_cost_USD: Decimal = Decimal(0)
    reasoning_cost_USD: Decimal = Decimal(0)
    audio_input_cost_USD: Decimal = Decimal(0)
    audio_output_cost_USD: Decimal = Decimal(0)
    image_input_cost_USD: Decimal = Decimal(0)
    total_cost_USD: Decimal = Decimal(0)

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def __str__(self) -> str:
        def fmt(value: Decimal | None) -> str:
            return f"{round(value, 2)}"
        
        lines = ["Cost breakdown:"]
        if self.input_cost_USD:
            lines.append(f"  Input:          ${fmt(self.input_cost_USD)}")
        if self.output_cost_USD:
            lines.append(f"  Output:         ${fmt(self.output_cost_USD)}")
        if self.cached_input_cost_USD:
            lines.append(f"  Cached input:   ${fmt(self.cached_input_cost_USD)}")
        if self.cache_creation_cost_USD:
            lines.append(f"  Cache creation: ${fmt(self.cache_creation_cost_USD)}")
        if self.cache_read_cost_USD:
            lines.append(f"  Cache read:     ${fmt(self.cache_read_cost_USD)}")
        if self.reasoning_cost_USD:
            lines.append(f"  Reasoning:      ${fmt(self.reasoning_cost_USD)}")
        if self.audio_input_cost_USD:
            lines.append(f"  Audio input:    ${fmt(self.audio_input_cost_USD)}")
        if self.audio_output_cost_USD:
            lines.append(f"  Audio output:   ${fmt(self.audio_output_cost_USD)}")
        if self.image_input_cost_USD:
            lines.append(f"  Image input:    ${fmt(self.image_input_cost_USD)}")
        if self.total_cost_USD:
            lines.append(f"  Total:          ${fmt(self.total_cost_USD)}")
        return "\n".join(lines)
    
@dataclass
class Usage:
    tokens: UsageBreakdown = field(default_factory=UsageBreakdown)
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    def __add__(self, other: Usage) -> Usage:
        return Usage(tokens=self.tokens + other.tokens, cost=self.cost + other.cost)

    def __str__(self) -> str:
        return f"{self.tokens}\n{self.cost}"


def _get_cost(model: ModelConfig, usage: UsageBreakdown) -> CostBreakdown:
    pricing: ModelPricing = model.pricing
    costs = CostBreakdown()

    if model.provider == "openai":
        non_cached_input_tokens = usage.input_tokens - usage.cached_tokens
        costs.input_cost_USD = (Decimal(non_cached_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.cached_input_cost_USD = (Decimal(usage.cached_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cached_input_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD + costs.cached_input_cost_USD
    
    elif model.provider == "anthropic":
        costs.input_cost_USD = (Decimal(usage.input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.cache_creation_cost_USD = (Decimal(usage.cache_creation_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cache_creation_input_per_1m))
        costs.cache_read_cost_USD = (Decimal(usage.cache_read_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.cache_read_input_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD + costs.cache_creation_cost_USD + costs.cache_read_cost_USD
    
    elif model.provider == "gemini":
        costs.input_cost_USD = (Decimal(usage.input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.input_per_1m))
        costs.output_cost_USD = (Decimal(usage.output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing.output_per_1m))
        costs.total_cost_USD = costs.input_cost_USD + costs.output_cost_USD

    if costs.total_cost_USD is None:
        raise ValueError("Total cost could not be calculated due to missing usage or pricing information")
    
    return costs

def get_usage(model: ModelConfig, usage: UsageBreakdown) -> Usage:
    cost = _get_cost(model, usage)
    return Usage(tokens=usage, cost=cost)

