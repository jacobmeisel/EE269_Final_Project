function names = getLabelNames(labels)
   labelNames = {'Normal (N)';
                 'Supraventricular premature beat (S)';
                 'Premature ventricular contraction (V)';
                 'Fusion of ventricular and normal beat (F)';
                 'Unclassifiable beat (Q)'};
   names = labelNames(labels + 1);
end