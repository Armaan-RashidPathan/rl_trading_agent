"""
Cross-Modal Attention Module
Learns dynamic relationships between price and sentiment features
Provides interpretable attention weights for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for Price-Sentiment Fusion

    Architecture:
        1. Embed price features → 64 dims
        2. Embed sentiment features → 64 dims
        3. Compute cross-attention between modalities
        4. Fuse attended representations
        5. Output unified state representation

    Key Feature:
        Attention weights are stored for explainability
        Shows how much sentiment influenced price decisions
    """

    def __init__(
        self,
        price_dim,
        sentiment_dim,
        hidden_dim=64,
        output_dim=128
    ):
        """
        Args:
            price_dim:     Number of price/technical features
            sentiment_dim: Number of sentiment features
            hidden_dim:    Embedding dimension
            output_dim:    Output dimension
        """
        super(CrossModalAttention, self).__init__()

        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim

        # Price embedding network
        self.price_embed = nn.Sequential(
            nn.Linear(price_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Sentiment embedding network
        self.sentiment_embed = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Attention projection layers
        # Price attending to Sentiment
        self.query_price = nn.Linear(hidden_dim, hidden_dim)
        self.key_sent    = nn.Linear(hidden_dim, hidden_dim)
        self.value_sent  = nn.Linear(hidden_dim, hidden_dim)

        # Sentiment attending to Price
        self.query_sent  = nn.Linear(hidden_dim, hidden_dim)
        self.key_price   = nn.Linear(hidden_dim, hidden_dim)
        self.value_price = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        # Input: price_embed + attended_price + sent_embed + attended_sent
        # = 4 × hidden_dim = 4 × 64 = 256
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        # Store attention weights for explainability
        self.price_to_sent_attn  = None
        self.sent_to_price_attn  = None

        # Scale factor for attention
        self.scale = np.sqrt(hidden_dim)

    def forward(self, price_features, sentiment_features):
        """
        Forward pass

        Args:
            price_features:     Tensor [batch, price_dim]
            sentiment_features: Tensor [batch, sentiment_dim]

        Returns:
            fused: Tensor [batch, output_dim]
        """
        # Step 1: Embed both modalities
        P = self.price_embed(price_features)       # [batch, hidden]
        S = self.sentiment_embed(sentiment_features) # [batch, hidden]

        # Step 2: Price → Sentiment Attention
        # "How much should price features attend to sentiment?"
        Q_p  = self.query_price(P)  # [batch, hidden]
        K_s  = self.key_sent(S)     # [batch, hidden]
        V_s  = self.value_sent(S)   # [batch, hidden]

        # Attention score
        attn_score_ps = torch.sum(Q_p * K_s, dim=-1, keepdim=True) / self.scale
        attn_weight_ps = torch.sigmoid(attn_score_ps)  # [batch, 1]

        # Attended sentiment (weighted by attention)
        P_attended = attn_weight_ps * V_s  # [batch, hidden]

        # Store for explainability
        self.price_to_sent_attn = attn_weight_ps.detach()

        # Step 3: Sentiment → Price Attention
        # "How much should sentiment features attend to price?"
        Q_s  = self.query_sent(S)   # [batch, hidden]
        K_p  = self.key_price(P)    # [batch, hidden]
        V_p  = self.value_price(P)  # [batch, hidden]

        # Attention score
        attn_score_sp = torch.sum(Q_s * K_p, dim=-1, keepdim=True) / self.scale
        attn_weight_sp = torch.sigmoid(attn_score_sp)  # [batch, 1]

        # Attended price (weighted by attention)
        S_attended = attn_weight_sp * V_p  # [batch, hidden]

        # Store for explainability
        self.sent_to_price_attn = attn_weight_sp.detach()

        # Step 4: Fuse all representations
        # Concatenate: original + attended for both modalities
        fused = torch.cat([P, P_attended, S, S_attended], dim=-1)  # [batch, 256]

        # Step 5: Project to output dimension
        output = self.output_proj(fused)  # [batch, output_dim]

        return output

    def get_attention_weights(self):
        """
        Get stored attention weights for explainability

        Returns:
            dict with attention weights
        """
        weights = {}

        if self.price_to_sent_attn is not None:
            weights['price_to_sentiment'] = (
                self.price_to_sent_attn.cpu().numpy()
            )

        if self.sent_to_price_attn is not None:
            weights['sentiment_to_price'] = (
                self.sent_to_price_attn.cpu().numpy()
            )

        return weights


class AttentionFusionLayer(nn.Module):
    """
    Complete fusion layer that:
    1. Splits features into price and sentiment
    2. Applies cross-modal attention
    3. Returns fused representation

    Used directly by the PPO agent
    """

    def __init__(self, total_features=24, output_dim=128):
        """
        Args:
            total_features: Total input features (24)
            output_dim:     Output dimension for RL agent
        """
        super(AttentionFusionLayer, self).__init__()

        # Feature split
        # Technical + Regime = 19 features (price-related)
        # Sentiment = 5 features
        self.price_dim    = 19
        self.sentiment_dim = 5

        # Cross-modal attention
        self.attention = CrossModalAttention(
            price_dim=self.price_dim,
            sentiment_dim=self.sentiment_dim,
            hidden_dim=64,
            output_dim=output_dim
        )

        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Combined feature tensor [batch, 24]

        Returns:
            fused: [batch, output_dim]
        """
        # Split features
        price_features     = x[:, :self.price_dim]       # First 19
        sentiment_features = x[:, self.price_dim:]       # Last 5

        # Apply cross-modal attention
        fused = self.attention(price_features, sentiment_features)

        return fused

    def get_attention_weights(self):
        """Get attention weights from inner module"""
        return self.attention.get_attention_weights()


def test_attention():
    """Test the cross-modal attention module"""

    print("=" * 50)
    print("TESTING CROSS-MODAL ATTENTION")
    print("=" * 50)

    # Test dimensions
    batch_size    = 4
    price_dim     = 19
    sentiment_dim = 5
    total_dim     = price_dim + sentiment_dim  # 24

    print(f"\nInput dimensions:")
    print(f"  Price features:     {price_dim}")
    print(f"  Sentiment features: {sentiment_dim}")
    print(f"  Total features:     {total_dim}")

    # Create dummy data
    price_data     = torch.randn(batch_size, price_dim)
    sentiment_data = torch.randn(batch_size, sentiment_dim)
    combined_data  = torch.randn(batch_size, total_dim)

    # Test CrossModalAttention
    print("\n[1/2] Testing CrossModalAttention...")
    attention = CrossModalAttention(
        price_dim=price_dim,
        sentiment_dim=sentiment_dim,
        hidden_dim=64,
        output_dim=128
    )

    output = attention(price_data, sentiment_data)
    print(f"  Input:  price {price_data.shape} + sentiment {sentiment_data.shape}")
    print(f"  Output: {output.shape}")
    print(f"  ✓ CrossModalAttention working")

    # Get attention weights
    weights = attention.get_attention_weights()
    print(f"\n  Attention Weights:")
    for name, weight in weights.items():
        print(f"    {name}: {weight.shape} | mean: {weight.mean():.3f}")

    # Test AttentionFusionLayer
    print("\n[2/2] Testing AttentionFusionLayer...")
    fusion = AttentionFusionLayer(
        total_features=total_dim,
        output_dim=128
    )

    fused = fusion(combined_data)
    print(f"  Input:  {combined_data.shape}")
    print(f"  Output: {fused.shape}")
    print(f"  ✓ AttentionFusionLayer working")

    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # Test with real data
    print("\n[3/3] Testing with real feature data...")
    import pandas as pd
    import yaml

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    features = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0,
        parse_dates=True
    )

    # Convert first row to tensor
    sample = torch.FloatTensor(
        features.iloc[:4].values
    )

    fused_real = fusion(sample)
    print(f"  Real features shape: {sample.shape}")
    print(f"  Fused output shape:  {fused_real.shape}")
    print(f"  Output range: [{fused_real.min():.3f}, {fused_real.max():.3f}]")
    print(f"  ✓ Works with real data!")

    # Show attention weights on real data
    weights = fusion.get_attention_weights()
    print(f"\n  Attention weights on real data:")
    for name, weight in weights.items():
        print(f"    {name}: mean={weight.mean():.3f}")

    print("\n" + "=" * 50)
    print("CROSS-MODAL ATTENTION TEST COMPLETE!")
    print("Next: Integration Testing")
    print("=" * 50)

    return fusion


if __name__ == "__main__":
    test_attention()