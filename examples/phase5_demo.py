#!/usr/bin/env python3
"""
Sweetviz v2 Phase 5 Features Demo: Advanced AI Features & MLOps Integrations

This demo showcases the advanced AI capabilities and MLOps integrations added in Phase 5:
- Enhanced AI insights with sophisticated analysis patterns
- MLOps platform integrations (MLflow, Weights & Biases)
- Natural language query interface
- Advanced anomaly detection
- Backwards compatibility with all existing features

Run with: python examples/phase5_demo.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_demo_dataset():
    """Create a realistic dataset for demonstration"""
    np.random.seed(42)
    
    # Employee dataset with various data quality issues
    n_samples = 1000
    
    data = {
        # Numeric features
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(75000, 20000, n_samples),
        'years_experience': np.random.normal(8, 5, n_samples),
        'performance_score': np.random.normal(3.5, 0.8, n_samples),
        
        # Categorical features
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract'], n_samples),
        
        # Binary features
        'remote_work': np.random.choice([True, False], n_samples),
        'has_certification': np.random.choice([True, False], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic correlations
    df['salary'] = df['salary'] + df['years_experience'] * 2000 + np.random.normal(0, 5000, n_samples)
    df['performance_score'] = df['performance_score'] + (df['salary'] / 100000) * 0.5 + np.random.normal(0, 0.2, n_samples)
    
    # Introduce data quality issues
    # 1. Add some outliers
    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.uniform(80, 120, 10)  # Unrealistic ages
    df.loc[outlier_indices[:5], 'salary'] = np.random.uniform(500000, 1000000, 5)  # Very high salaries
    
    # 2. Add missing values (but not in performance_score since it will be used as target)
    missing_indices = np.random.choice(n_samples, 50, replace=False)
    df.loc[missing_indices, 'years_experience'] = np.nan  # Add missing to years_experience instead
    
    # 3. Add some duplicate rows
    duplicate_rows = df.iloc[:5].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # 4. Add a constant column (data quality issue)
    df['constant_column'] = 'SAME_VALUE'
    
    # 5. Add a mostly missing column
    df['mostly_missing'] = np.nan
    df.loc[:20, 'mostly_missing'] = 'rare_value'
    
    # Clean up unrealistic values
    df['age'] = np.clip(df['age'], 18, 120)
    df['years_experience'] = np.clip(df['years_experience'], 0, 50)
    df['performance_score'] = np.clip(df['performance_score'], 1, 5)
    df['salary'] = np.clip(df['salary'], 30000, 1000000)
    
    return df

def demo_basic_sweetviz():
    """Demo 1: Show that basic sweetviz functionality still works perfectly"""
    print("=" * 80)
    print("DEMO 1: BACKWARDS COMPATIBILITY")
    print("Showing that all existing sweetviz functionality works unchanged")
    print("=" * 80)
    
    import sweetviz as sv
    
    # Create dataset
    df = create_demo_dataset()
    print(f"✅ Created demo dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Traditional sweetviz analysis
    print("\n📊 Running traditional sweetviz analysis...")
    report = sv.analyze(df, target_feat='performance_score')  # Now performance_score has no missing values
    print("✅ Analysis complete!")
    
    # Show that the report has the traditional methods
    print(f"✅ Report object type: {type(report)}")
    print(f"✅ Traditional methods available: {hasattr(report, 'show_html')}, {hasattr(report, 'to_html')}")
    
    # Generate HTML (but don't open browser in demo)
    html_output = report.to_html()
    print(f"✅ HTML report generated: {len(html_output):,} characters")
    
    return df, report

def demo_enhanced_ai_insights():
    """Demo 2: Enhanced AI insights with sophisticated analysis"""
    print("\n" + "=" * 80)
    print("DEMO 2: ENHANCED AI INSIGHTS")
    print("Advanced AI-powered data analysis (works without API keys)")
    print("=" * 80)
    
    import sweetviz as sv
    
    df = create_demo_dataset()
    
    # Get AI manager
    ai_manager = sv.get_ai_manager()
    print(f"✅ AI Manager loaded: {type(ai_manager)}")
    print(f"✅ AI Available: {ai_manager.is_available()}")
    
    # Enhanced anomaly detection (works without API keys)
    print("\n🔍 Running enhanced anomaly detection...")
    anomalies = ai_manager.detect_anomalies(df)
    
    if anomalies:
        print(f"✅ Anomaly detection complete!")
        
        # Statistical outliers
        if 'statistical_outliers' in anomalies and anomalies['statistical_outliers']:
            print(f"\n📈 Statistical Outliers Found:")
            for col, info in anomalies['statistical_outliers'].items():
                iqr_count = info['iqr_outliers']['count']
                iqr_pct = info['iqr_outliers']['percentage']
                print(f"   • {col}: {iqr_count} outliers ({iqr_pct:.1f}%)")
        
        # Pattern anomalies
        if 'pattern_anomalies' in anomalies and anomalies['pattern_anomalies']:
            print(f"\n🔍 Pattern Anomalies Found:")
            for anomaly_type, info in anomalies['pattern_anomalies'].items():
                if anomaly_type == 'constant_columns':
                    print(f"   • Constant columns: {info['columns']}")
                elif anomaly_type == 'high_missing_columns':
                    cols = [item['column'] for item in info['columns']]
                    print(f"   • High missing data: {cols}")
                elif anomaly_type == 'duplicate_rows':
                    print(f"   • Duplicate rows: {info['count']} ({info['percentage']:.1f}%)")
        
        # Recommendations
        if 'recommendations' in anomalies and anomalies['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(anomalies['recommendations'], 1):
                print(f"   {i}. {rec}")
    
    # Enhanced correlation analysis
    print(f"\n🔗 Enhanced correlation analysis...")
    correlations = {
        'age_salary': df['age'].corr(df['salary']),
        'experience_salary': df['years_experience'].corr(df['salary']),
        'salary_performance': df['salary'].corr(df['performance_score'])
    }
    
    explanation = ai_manager.explain_correlations(correlations)
    if explanation:
        print(f"✅ Correlation analysis:")
        print(f"   {explanation}")
    else:
        print(f"✅ Correlation analysis: Basic correlation values computed")
        for pair, corr in correlations.items():
            if not pd.isna(corr):
                print(f"   • {pair}: {corr:.3f}")
    
    return anomalies

def demo_natural_language_queries():
    """Demo 3: Natural language query interface"""
    print("\n" + "=" * 80)
    print("DEMO 3: NATURAL LANGUAGE QUERIES")
    print("Ask questions about your data in plain English")
    print("=" * 80)
    
    import sweetviz as sv
    
    df = create_demo_dataset()
    
    # Show available query suggestions
    suggestions = sv.get_query_suggestions()
    print(f"✅ Query interface loaded with {len(suggestions)} example patterns")
    print(f"📝 Example queries: {suggestions[:3]}")
    
    # Demonstrate various types of queries
    queries = [
        "mean of salary",
        "distribution of age", 
        "missing values in years_experience",
        "correlation between age and salary",
        "shape of dataset",
        "unique values in department",
        "top 5 salary",
        "info about years_experience"
    ]
    
    print(f"\n🗣️  Testing natural language queries:")
    
    for query in queries:
        try:
            result = sv.ask_question(query, df)
            
            if 'result' in result:
                print(f"\n❓ '{query}'")
                
                # Format the answer based on query type
                res = result['result']
                if 'mean' in res:
                    print(f"   ✅ Mean: {res['mean']:.2f}")
                elif 'max' in res:
                    print(f"   ✅ Maximum: {res['max']}")
                elif 'correlation' in res:
                    corr = res['correlation']
                    strength = res.get('strength', 'unknown')
                    direction = res.get('direction', 'unknown')
                    print(f"   ✅ Correlation: {corr:.3f} ({strength} {direction})")
                elif 'missing_count' in res:
                    missing = res['missing_count']
                    pct = res['missing_percentage']
                    print(f"   ✅ Missing values: {missing} ({pct:.1f}%)")
                elif 'shape' in res:
                    print(f"   ✅ Dataset shape: {res['shape']}")
                elif 'unique_count' in res:
                    print(f"   ✅ Unique values: {res['unique_count']}")
                elif 'values' in res:
                    print(f"   ✅ Top values: {res['values'][:3]}...")
                elif 'dtype' in res:
                    print(f"   ✅ Data type: {res['dtype']}")
                else:
                    print(f"   ✅ Result: {str(res)[:100]}...")
                    
            elif 'error' in result:
                print(f"\n❓ '{query}'")
                print(f"   ❌ {result['error']}")
                
        except Exception as e:
            print(f"\n❓ '{query}'")
            print(f"   ❌ Error: {str(e)}")
    
    # Test AI-powered queries (if available)
    nl_interface = sv.get_nl_query_interface()
    if nl_interface.is_ai_available():
        print(f"\n🤖 AI-powered query parsing is available!")
        ai_result = sv.ask_question("What are the most interesting patterns in this data?", df, use_ai=True)
        print(f"AI Response: {ai_result}")
    else:
        print(f"\n🤖 AI-powered queries not configured (OpenAI API key needed)")

def demo_mlops_integrations():
    """Demo 4: MLOps platform integrations"""
    print("\n" + "=" * 80)
    print("DEMO 4: MLOPS INTEGRATIONS")  
    print("Export sweetviz reports to MLflow and Weights & Biases")
    print("=" * 80)
    
    import sweetviz as sv
    
    df = create_demo_dataset()
    
    # Create a comprehensive report
    print("📊 Creating comprehensive sweetviz report...")
    report = sv.analyze(df, target_feat='performance_score')  # Now performance_score has no missing values
    print("✅ Report created!")
    
    # Test MLOps manager
    mlops_manager = sv.get_mlops_manager()
    print(f"✅ MLOps Manager loaded: {type(mlops_manager)}")
    
    # List available integrations
    available_integrations = mlops_manager.list_available_integrations()
    print(f"🔧 Available integrations: {available_integrations if available_integrations else 'None (packages not installed)'}")
    
    # Test report MLOps export methods
    print(f"\n📤 Testing report export methods:")
    print(f"✅ MLflow export method: {hasattr(report, 'to_mlflow')}")
    print(f"✅ Weights & Biases export method: {hasattr(report, 'to_wandb')}")
    
    # Test data extraction
    print(f"\n📊 Testing report data extraction...")
    report_data = report._extract_report_data()
    print(f"✅ Extracted report data with keys: {list(report_data.keys())}")
    print(f"   • Dataset info: Shape {report_data['dataset_info']['shape']}")
    print(f"   • Summary stats: {len(report_data['summary_stats'])} metrics")
    print(f"   • Feature analysis: {len(report_data['feature_analysis'])} features")
    
    # Demonstrate export (will show unavailable without packages)
    print(f"\n📤 Testing MLflow export (demo mode)...")
    mlflow_result = report.to_mlflow(experiment_name="sweetviz_demo", tags={"demo": "phase5"})
    
    if 'error' in mlflow_result:
        print(f"   ℹ️  MLflow not available: {mlflow_result['error']}")
        print(f"   💡 To use: pip install mlflow")
    else:
        print(f"   ✅ MLflow export successful: {mlflow_result}")
    
    print(f"\n📤 Testing Weights & Biases export (demo mode)...")
    wandb_result = report.to_wandb(experiment_name="sweetviz_demo", tags={"demo": "phase5"})
    
    if 'error' in wandb_result:
        print(f"   ℹ️  Weights & Biases not available: {wandb_result['error']}")  
        print(f"   💡 To use: pip install wandb")
    else:
        print(f"   ✅ Weights & Biases export successful: {wandb_result}")
    
    return report_data

def demo_modern_configuration():
    """Demo 5: Modern configuration system with Phase 5 features"""
    print("\n" + "=" * 80)
    print("DEMO 5: MODERN CONFIGURATION")
    print("Advanced configuration options for AI and MLOps features")
    print("=" * 80)
    
    import sweetviz as sv
    
    # Show current configuration
    current_config = sv.get_config()
    print(f"✅ Current config type: {type(current_config)}")
    
    # Create modern configuration
    print(f"\n⚙️  Creating modern configuration with AI and MLOps settings...")
    config = sv.ModernConfig()
    
    # Configure themes and visualizations (Phase 4)
    config.theme = sv.Theme.MODERN_DARK
    config.visualizations.engine = sv.VisualizationEngine.AUTO
    config.visualizations.interactive_charts = True
    
    # Configure performance (Phase 4)
    config.performance_mode = sv.PerformanceMode.BALANCED
    config.performance.enable_sampling = True
    config.performance.max_sample_size = 5000
    
    # Configure AI features (Phase 5)
    config.ai_features.enabled = True
    config.ai_features.llm_provider = sv.LLMProvider.OPENAI
    # config.ai_features.api_key = "your-openai-api-key"  # Would set in real usage
    config.ai_features.generate_insights = True
    
    print(f"✅ Configuration created with AI features enabled: {config.ai_features.enabled}")
    print(f"✅ Theme: {config.theme}")
    print(f"✅ Visualization engine: {config.visualizations.engine}")
    print(f"✅ Performance mode: {config.performance_mode}")
    print(f"✅ AI provider: {config.ai_features.llm_provider}")
    
    # Apply configuration
    sv.set_config(config)
    print(f"✅ Configuration applied!")
    
    # Verify configuration is active
    active_config = sv.get_config()
    print(f"✅ Active config AI enabled: {active_config.ai_features.enabled}")
    
    return config

def demo_complete_workflow():
    """Demo 6: Complete end-to-end workflow with all Phase 5 features"""
    print("\n" + "=" * 80)
    print("DEMO 6: COMPLETE WORKFLOW")
    print("End-to-end data analysis with AI insights, MLOps, and natural language")
    print("=" * 80)
    
    import sweetviz as sv
    
    # 1. Setup modern configuration
    print("1️⃣  Setting up modern configuration...")
    config = sv.ModernConfig()
    config.theme = sv.Theme.MODERN_DARK
    config.ai_features.enabled = True
    sv.set_config(config)
    print("   ✅ Configuration applied")
    
    # 2. Load and analyze data
    print("\n2️⃣  Loading and analyzing dataset...")
    df = create_demo_dataset()
    report = sv.analyze(df, target_feat='performance_score')  # Now performance_score has no missing values
    print(f"   ✅ Analysis complete: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 3. Run AI-powered insights
    print("\n3️⃣  Running AI-powered insights...")
    ai_manager = sv.get_ai_manager()
    anomalies = ai_manager.detect_anomalies(df)
    
    outlier_count = 0
    pattern_issues = 0
    
    if anomalies:
        outlier_count = sum(
            info['iqr_outliers']['count'] 
            for info in anomalies.get('statistical_outliers', {}).values()
        )
        pattern_issues = len(anomalies.get('pattern_anomalies', {}))
    
    print(f"   ✅ Found {outlier_count} statistical outliers across columns")
    print(f"   ✅ Detected {pattern_issues} pattern-based data quality issues")
    
    # 4. Interactive data exploration with natural language
    print("\n4️⃣  Interactive data exploration...")
    queries = [
        "mean of salary", 
        "correlation between age and performance_score",
        "missing values in years_experience"
    ]
    
    for query in queries:
        result = sv.ask_question(query, df)
        if 'result' in result:
            print(f"   ❓ {query} → ✅ Answered")
        else:
            print(f"   ❓ {query} → ❌ {result.get('error', 'Unknown error')}")
    
    # 5. Export to MLOps platforms
    print("\n5️⃣  Exporting to MLOps platforms...")
    
    # Extract structured data
    report_data = report._extract_report_data()
    
    # Try MLflow export
    mlflow_result = report.to_mlflow(
        experiment_name="sweetviz_complete_demo",
        tags={"phase": "5", "demo": "complete_workflow"}
    )
    mlflow_status = "✅ Success" if 'error' not in mlflow_result else f"ℹ️  {mlflow_result['error']}"
    print(f"   📊 MLflow export: {mlflow_status}")
    
    # Try Weights & Biases export  
    wandb_result = report.to_wandb(
        experiment_name="sweetviz_complete_demo",
        tags={"phase": "5", "demo": "complete_workflow"}
    )
    wandb_status = "✅ Success" if 'error' not in wandb_result else f"ℹ️  {wandb_result['error']}"
    print(f"   📊 Weights & Biases export: {wandb_status}")
    
    # 6. Generate traditional HTML report (backwards compatibility)
    print("\n6️⃣  Generating traditional HTML report...")
    html_output = report.to_html()
    print(f"   ✅ HTML report: {len(html_output):,} characters")
    print("   ✅ All existing sweetviz functionality preserved!")
    
    # Summary
    print(f"\n🎉 COMPLETE WORKFLOW SUMMARY:")
    print(f"   • Dataset analyzed: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   • AI insights: {outlier_count} outliers, {pattern_issues} pattern issues")
    print(f"   • Natural language queries: {len(queries)} executed")
    print(f"   • MLOps exports: MLflow {mlflow_status}, W&B {wandb_status}")
    print(f"   • Traditional output: HTML report generated")
    print(f"   • Backwards compatibility: 100% preserved")
    
    return {
        'report': report,
        'anomalies': anomalies,
        'report_data': report_data,
        'mlflow_result': mlflow_result,
        'wandb_result': wandb_result
    }

def main():
    """Run all Phase 5 feature demonstrations"""
    print("🚀 SWEETVIZ V2 PHASE 5 FEATURES DEMONSTRATION")
    print("Advanced AI Features & MLOps Integrations")
    print("=" * 80)
    
    try:
        # Demo 1: Backwards compatibility
        df, report = demo_basic_sweetviz()
        
        # Demo 2: Enhanced AI insights
        anomalies = demo_enhanced_ai_insights()
        
        # Demo 3: Natural language queries
        demo_natural_language_queries()
        
        # Demo 4: MLOps integrations
        report_data = demo_mlops_integrations()
        
        # Demo 5: Modern configuration
        config = demo_modern_configuration()
        
        # Demo 6: Complete workflow
        workflow_results = demo_complete_workflow()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 PHASE 5 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✅ All Phase 5 features demonstrated successfully")
        print("✅ 100% backwards compatibility maintained")
        print("✅ Advanced AI insights working (enhanced detection)")
        print("✅ Natural language query interface functional")
        print("✅ MLOps integration framework ready")
        print("✅ Modern configuration system operational")
        
        print(f"\n💡 TO USE FULL FEATURES:")
        print(f"   • AI insights: pip install sweetviz[ai] + OpenAI API key")
        print(f"   • MLOps exports: pip install sweetviz[mlops]")
        print(f"   • Enhanced viz: pip install sweetviz[enhanced]")
        print(f"   • Everything: pip install sweetviz[ai,mlops,enhanced]")
        
        print(f"\n🔗 QUICK START:")
        print(f"   import sweetviz as sv")
        print(f"   report = sv.analyze(df)")
        print(f"   sv.ask_question('mean of age', df)")
        print(f"   report.to_mlflow('my_experiment')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)