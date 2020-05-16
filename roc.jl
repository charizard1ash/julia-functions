using DataFrames
df = DataFrames

function roc(prob_values, labels; prob_thresh = collect(0:0.05:1))
    df_roc = df.DataFrame(prob_threshold = Float64[], tnr = Float64[], fpr = Float64[], fnr = Float64[], tpr = Float64[], precision = Float64[], recall = Float64[], f1_score = Float64[])
    for p in prob_thresh
        x_pred = ifelse.(prob_values>=p, 1, 0)
        df_2 = df.DataFrame(pred=x_pred, actual=labels)
        tnr = df.nrow(df_2[(df_2[:pred] .== 0) .& (df_2[:actual] .== 0), :]) / df.nrow(df_2[df_2[:actual] .== 0, :])
        fpr = df.nrow(df_2[(df_2[:pred] .== 1) .& (df_2[:actual] .== 0), :]) / df.nrow(df_2[df_2[:actual] .== 0, :])
        fnr = df.nrow(df_2[(df_2[:pred] .== 0) .& (df_2[:actual] .== 1), :]) / df.nrow(df_2[df_2[:actual] .== 1, :])
        tpr = df.nrow(df_2[(df_2[:pred] .== 1) .& (df_2[:actual] .== 1), :]) / df.nrow(df_2[df_2[:actual] .== 1, :])
        precision = df.nrow(df_2[(df_2[:pred] .== 1) .& (df_2[:actual] .== 1), :]) / df.nrow(df_2[df_2[:pred] .== 1, :])
        recall = df.nrow(df_2[(df_2[:pred] .== 1) .& (df_2[:actual] .== 1), :]) / df.nrow(df_2[df_2[:actual] .== 1, :])
        f1_score = (2 * precision * recall) / (precision + recall)
        push!(df_roc, [p tnr fpr fnr tpr precision recall f1_score])
    end
    
    auc = unique(df_roc[:, [:fpr, :tpr]])
    sort!(auc, order([:fpr, :tpr]), rev=(false, false))
    auc[!, :fpr_prev] = [missing, auc[1:(df.nrow(auc)-1), :fpr]]
    auc[!, :tpr_prev] = [missing, auc[1:(df.nrow(auc)-1), :tpr]]
    auc[!, :tpr_min] = ifelse.(abs.(auc[:tpr_prev]) .< abs.(auc[:tpr]), auc[:tpr_prev], auc[:tpr])
    auc[!, :tpr_max] = ifelse.(abs.(auc[:tpr_prev]) .>= abs.(auc[:tpr]), auc[:tpr_prev], auc[:tpr])
    auc[!, :area] = map((x2, x1, y2, y1) -> (x2 - x1) * (y1 + 0.5 * (y2 - y1)), auc[:fpr], auc[:fpr_prev], auc[:tpr_max], auc[:tpr_min])
    auc = sum(auc[ismissing.(auc[:area]) .== false, :area])
    
    return df_roc, auc
end
