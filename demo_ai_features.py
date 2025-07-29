#!/usr/bin/env python3
"""
Sweetviz v2 AI Features Demonstration

This script demonstrates the new AI-powered features in Sweetviz v2,
including automated insights, smart data type detection, and anomaly detection.
"""

import pandas as pd
import numpy as np
import sweetviz as sv
from sweetviz.ai_insights import DataInsightGenerator, SmartDataDetection
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for demo clarity


def create_demo_dataset():
    """Create a realistic dataset for demonstration."""
    print("🔧 Creating demonstration dataset...")
    
    np.random.seed(42)
    n = 500
    
    # Create realistic data with various patterns
    data = {
        # Identifiers and contact info
        'customer_id': [f'CUST{i:05d}' for i in range(n)],
        'email': [f'user{i}@{"premium" if i%10==0 else "standard"}.com' for i in range(n)],
        'phone': [f'({np.random.randint(200,999):03d}) {np.random.randint(100,999):03d}-{np.random.randint(1000,9999):04d}' for _ in range(n)],
        
        # Demographics
        'age': np.random.normal(38, 15, n).astype(int),
        'income': np.random.lognormal(10.5, 0.7, n),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n),
        
        # Behavioral data
        'signup_date': pd.date_range('2020-01-01', periods=n, freq='D')[:n],
        'is_premium': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'active_flag': np.random.choice(['Y', 'N'], n, p=[0.8, 0.2]),
        'loyalty_score': np.random.beta(2, 5, n) * 100,
        
        # Transaction data with some correlation to income
        'monthly_spend': np.random.exponential(50, n) + (np.random.normal(0, 20, n)),
        'total_orders': np.random.poisson(5, n),
    }
    
    df = pd.DataFrame(data)
    
    # Make monthly_spend correlated with income and premium status
    df['monthly_spend'] = (df['income'] / 1000 + 
                          df['is_premium'] * 50 + 
                          np.random.normal(0, 30, n)).clip(0)
    
    # Add some missing values (realistic pattern)
    missing_indices = np.random.choice(n, int(n * 0.05), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n, 10, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.choice([15, 95, 120], 10)  # Age outliers
    df.loc[outlier_indices[:5], 'monthly_spend'] = np.random.uniform(5000, 10000, 5)  # Spend outliers
    
    print(f"✅ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def demonstrate_basic_ai_features(df):
    """Demonstrate basic AI features integrated into standard workflow."""
    print("\n" + "="*80)
    print("🤖 DEMONSTRATION 1: BASIC AI INTEGRATION")
    print("="*80)
    print("Running standard sweetviz analyze() with AI enhancements...")
    
    # Standard analyze call - AI insights are automatically generated
    report = sv.analyze(df, target_feat='is_premium')
    
    print("\n📊 Standard report generated with AI insights automatically included!")
    
    # Access AI insights programmatically
    ai_insights = report.get_ai_insights()
    semantic_types = report.get_semantic_types()
    
    print(f"\n🧠 AI Analysis Results:")
    print(f"   • {len(ai_insights)} insight categories generated")
    print(f"   • {len(semantic_types)} columns analyzed for semantic types")
    
    return report


def demonstrate_semantic_detection(df):
    """Demonstrate advanced semantic type detection."""
    print("\n" + "="*80) 
    print("🏷️ DEMONSTRATION 2: SMART SEMANTIC TYPE DETECTION")
    print("="*80)
    
    detector = SmartDataDetection()
    semantic_types = detector.detect_semantic_types(df)
    
    print("🔍 Semantic Type Detection Results:")
    print("-" * 50)
    
    interesting_types = []
    for col_name, analysis in semantic_types.items():
        semantic_type = analysis['semantic_type']
        confidence = analysis['confidence']
        
        if semantic_type != 'unknown' and confidence > 0.6:
            status = "✅" if confidence > 0.8 else "🟡"
            print(f"{status} {col_name:15} → {semantic_type:15} (confidence: {confidence:.2f})")
            interesting_types.append((col_name, semantic_type, confidence))
        elif semantic_type == 'unknown':
            print(f"❓ {col_name:15} → {semantic_type:15} (confidence: {confidence:.2f})")
    
    print(f"\n🎯 Successfully detected {len(interesting_types)} semantic types with high confidence!")


def demonstrate_anomaly_detection(df):
    """Demonstrate anomaly detection capabilities."""
    print("\n" + "="*80)
    print("🚨 DEMONSTRATION 3: AI-POWERED ANOMALY DETECTION")
    print("="*80)
    
    generator = DataInsightGenerator(confidence_threshold=0.95)
    insights = generator.generate_insights(df)
    
    anomaly_results = insights['anomaly_detection']
    
    print("🔍 Anomaly Detection Results:")
    print("-" * 40)
    
    # Isolation Forest results
    if 'isolation_forest' in anomaly_results and 'error' not in anomaly_results['isolation_forest']:
        iso_results = anomaly_results['isolation_forest']
        anomaly_count = iso_results['anomaly_count']
        anomaly_percentage = iso_results['anomaly_percentage']
        
        print(f"🤖 Machine Learning (Isolation Forest):")
        print(f"   • {anomaly_count} anomalous rows detected")
        print(f"   • {anomaly_percentage:.1f}% of dataset flagged as anomalies")
        
        if anomaly_percentage > 5:
            print("   ⚠️  High anomaly rate - investigate data quality!")
        elif anomaly_percentage > 0:
            print("   ✅ Normal anomaly rate detected")
    
    # Statistical outlier detection
    if 'statistical_outliers' in anomaly_results:
        stat_outliers = anomaly_results['statistical_outliers']
        
        print(f"\n📊 Statistical Outlier Detection (IQR method):")
        for col, outlier_info in stat_outliers.items():
            outlier_count = outlier_info['outlier_count']
            outlier_percentage = outlier_info['outlier_percentage']
            
            if outlier_count > 0:
                status = "🔴" if outlier_percentage > 5 else "🟡" if outlier_percentage > 1 else "🟢"
                print(f"   {status} {col:15}: {outlier_count:3d} outliers ({outlier_percentage:.1f}%)")


def demonstrate_data_quality_assessment(df):
    """Demonstrate comprehensive data quality assessment."""
    print("\n" + "="*80)
    print("🏥 DEMONSTRATION 4: DATA QUALITY ASSESSMENT")
    print("="*80)
    
    generator = DataInsightGenerator()
    insights = generator.generate_insights(df)
    
    quality_metrics = insights['data_quality']
    
    print("📋 Data Quality Report:")
    print("-" * 30)
    
    # Basic metrics
    print(f"📏 Dataset Size: {quality_metrics['total_rows']:,} rows × {quality_metrics['total_columns']} columns")
    print(f"💾 Memory Usage: {quality_metrics['memory_usage_mb']:.1f} MB")
    print(f"❌ Missing Data: {quality_metrics['missing_data_percentage']:.1f}%")
    print(f"🔄 Duplicate Rows: {quality_metrics['duplicate_rows']} ({quality_metrics['duplicate_percentage']:.1f}%)")
    
    # Quality flags
    quality_flags = quality_metrics['quality_flags']
    if quality_flags:
        print(f"\n⚠️  Quality Alerts:")
        for flag in quality_flags:
            flag_descriptions = {
                'HIGH_MISSING_DATA': 'More than 10% of data is missing',
                'HIGH_DUPLICATE_RATE': 'More than 5% of rows are duplicates', 
                'LARGE_MEMORY_USAGE': 'Dataset uses more than 100MB of memory'
            }
            print(f"   🚩 {flag}: {flag_descriptions.get(flag, 'Unknown issue')}")
    else:
        print(f"\n✅ No data quality issues detected!")
    
    # Column-specific quality
    print(f"\n📊 Column Quality Analysis:")
    column_quality = quality_metrics['column_quality']
    
    for col, col_info in column_quality.items():
        missing_pct = col_info['missing_percentage']
        unique_pct = col_info['unique_percentage'] 
        
        if missing_pct > 10:
            status = "🔴"
        elif missing_pct > 0:
            status = "🟡"
        else:
            status = "🟢"
        
        print(f"   {status} {col:15}: {missing_pct:5.1f}% missing, {unique_pct:5.1f}% unique")


def demonstrate_statistical_insights(df):
    """Demonstrate automated statistical testing."""
    print("\n" + "="*80)
    print("📈 DEMONSTRATION 5: AUTOMATED STATISTICAL ANALYSIS")
    print("="*80)
    
    generator = DataInsightGenerator()
    insights = generator.generate_insights(df, target_col='is_premium')
    
    # Distribution insights
    dist_insights = insights['distribution_insights']
    print("📊 Distribution Analysis:")
    print("-" * 25)
    
    for col, analysis in dist_insights.items():
        dist_type = analysis['distribution_type']
        skewness = analysis['skewness']
        
        # Interpret skewness
        if abs(skewness) < 0.5:
            skew_desc = "symmetric"
        elif skewness > 0.5:
            skew_desc = "right-skewed"
        else:
            skew_desc = "left-skewed"
        
        print(f"   📏 {col:15}: {dist_type:20} ({skew_desc})")
    
    # Correlation insights
    corr_insights = insights['correlation_insights']
    if 'high_correlations' in corr_insights:
        high_corrs = corr_insights['high_correlations']
        
        if high_corrs:
            print(f"\n🔗 Strong Correlations Found:")
            print("-" * 30)
            for corr in high_corrs[:5]:  # Show top 5
                strength = corr['strength']
                f1, f2 = corr['feature1'], corr['feature2']
                corr_val = corr['correlation']
                print(f"   🔄 {f1:15} ↔ {f2:15}: r={corr_val:+.3f} ({strength})")
    
    # Target analysis
    if 'target_insights' in insights:
        target_insights = insights['target_insights']
        relationships = target_insights['feature_relationships']
        
        print(f"\n🎯 Target Relationship Analysis ('{target_insights['target_column']}'):")
        print("-" * 45)
        
        for rel in relationships[:5]:  # Show top 5
            feature = rel['feature']
            strength = rel['strength']
            significant = "✅" if rel['significant'] else "❌"
            
            print(f"   {significant} {feature:15}: strength={strength:.3f}")


def demonstrate_complete_workflow():
    """Demonstrate the complete modernized sweetviz workflow."""
    print("\n" + "="*80)
    print("🚀 DEMONSTRATION 6: COMPLETE MODERNIZED WORKFLOW")
    print("="*80)
    
    # Create two datasets for comparison
    df1 = create_demo_dataset()
    
    # Modify for comparison
    df2 = df1.copy()
    df2['income'] = df2['income'] * 1.15  # Increase income
    df2['age'] = df2['age'] + 3  # Increase age
    df2['monthly_spend'] = df2['monthly_spend'] * 1.2  # Increase spending
    
    print("\n🔄 Running complete comparison analysis with AI insights...")
    
    # Complete workflow with AI
    report = sv.compare(
        [df1, "Original Dataset"],
        [df2, "Modified Dataset"], 
        target_feat='is_premium'
    )
    
    print("\n✨ Complete analysis finished!")
    print("   • Traditional EDA completed")
    print("   • AI insights generated")
    print("   • Semantic types detected")
    print("   • Anomalies identified")
    print("   • Statistical tests performed")
    print("   • Target relationships analyzed")
    
    # Show that we can access all the AI features
    ai_insights = report.get_ai_insights()
    semantic_types = report.get_semantic_types()
    
    print(f"\n📊 Analysis Summary:")
    print(f"   • Dataset comparison: {df1.shape} vs {df2.shape}")
    print(f"   • AI insight categories: {len(ai_insights) if ai_insights else 0}")
    print(f"   • Semantic types detected: {len([t for t in semantic_types.values() if t['confidence'] > 0.7]) if semantic_types else 0}")
    
    return report


def main():
    """Run the complete demonstration."""
    print("🎉 SWEETVIZ V2 AI FEATURES DEMONSTRATION")
    print("="*80)
    print("This demonstration showcases the new AI-powered features")
    print("integrated into Sweetviz v2 while maintaining full backward compatibility.")
    print("="*80)
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Run all demonstrations
    demonstrate_basic_ai_features(df)
    demonstrate_semantic_detection(df)
    demonstrate_anomaly_detection(df)
    demonstrate_data_quality_assessment(df)
    demonstrate_statistical_insights(df)
    demonstrate_complete_workflow()
    
    print("\n" + "="*80)
    print("🎊 DEMONSTRATION COMPLETE!")
    print("="*80)
    print("Key Benefits of Sweetviz v2:")
    print("✅ Full backward compatibility with existing code")
    print("✅ AI-powered insights automatically generated")
    print("✅ Smart semantic type detection")
    print("✅ Advanced anomaly detection")
    print("✅ Comprehensive data quality assessment")
    print("✅ Automated statistical analysis")
    print("✅ Modern Python best practices")
    print("✅ Enhanced performance and reliability")
    print("\n🚀 Sweetviz v2 brings your EDA into the AI era!")
    print("="*80)


if __name__ == "__main__":
    main()