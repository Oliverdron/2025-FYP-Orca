from featureanalysis import FeatureAnalysis

def main():
    # Initialize feature analysis
    analyzer = FeatureAnalysis(data_dir='data', result_dir='result/feature_analysis')
    
    # Run complete analysis
    print("Starting feature analysis...")
    stats = analyzer.analyze_all_features()
    
    # Print basic statistics
    print("\nFeature Statistics:")
    print(stats)
    
    print("\nAnalysis complete. Results saved in result/feature_analysis/")

if __name__ == "__main__":
    main()