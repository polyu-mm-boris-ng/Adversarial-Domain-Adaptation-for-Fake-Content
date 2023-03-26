from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.spatial import distance
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable


def extract_feature(model_pth, data_iter, logger, cuda=True):
    model = torch.load(model_pth)
    model = model.eval()
    if cuda:
        model = model.cuda()

    final_output = []

    count = 0
    with torch.no_grad():
        for batch in data_iter:
            to_text = batch.Text

            if cuda:
                to_text = to_text.cuda()

            inputvo_text = Variable(to_text)

            x = inputvo_text.permute(1, 0)
            embedded = model.embedding.f_embed(x)
            embedded = embedded.unsqueeze(1)
            # try:
            conved = [
                F.relu(conv(embedded)).squeeze(3) for conv in model.feature.f_convs
            ]
            # except:
            #     continue
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            cat = model.feature.f_drop(torch.cat(pooled, dim=1))
            feature = cat.view(-1, 300)

            final_output.append(feature[0].tolist())
            count += 1

            if count % 100 == 0:
                logger.info(f"Extracted {count} features")

    return final_output


def cal_tran_score(model_no_da, model_da, source_out_iter, target_out_iter, logger, cuda):
    no_da_source_feature = extract_feature(model_no_da, source_out_iter, logger, cuda)
    no_da_target_feature = extract_feature(model_no_da, target_out_iter, logger, cuda)
    no_da_source_df = pd.DataFrame(no_da_source_feature).add_prefix("col_")
    no_da_source_centroid = no_da_source_df.mean(axis=0)
    no_da_target_df = pd.DataFrame(no_da_target_feature).add_prefix("col_")
    no_da_target_centroid = no_da_target_df.mean(axis=0)
    no_da_cos_sim = cosine_similarity([no_da_source_centroid], [no_da_target_centroid])[0]

    da_source_feature = extract_feature(model_da, source_out_iter, logger, cuda)
    da_target_feature = extract_feature(model_da, target_out_iter, logger, cuda)
    da_source_df = pd.DataFrame(da_source_feature).add_prefix("col_")
    da_source_centroid = da_source_df.mean(axis=0)
    da_target_df = pd.DataFrame(da_target_feature).add_prefix("col_")
    da_target_centroid = da_target_df.mean(axis=0)
    da_cos_sim = cosine_similarity([da_source_centroid], [da_target_centroid])[0]

    transferability_score = no_da_cos_sim / da_cos_sim

    logger.info(f"No DA source centroid: {no_da_source_centroid}")
    logger.info(f"No DA target centroid: {no_da_target_centroid}")
    logger.info(f"No DA cosine similarity: {no_da_cos_sim}")
    logger.info(f"DA source centroid: {da_source_centroid}")
    logger.info(f"DA target centroid: {da_target_centroid}")
    logger.info(f"DA cosine similarity: {da_cos_sim}")
    logger.info(f"Transferability score: {transferability_score}")

    return transferability_score, da_source_df, da_target_df, no_da_source_df, no_da_target_df

def visualize(da_source_df, da_target_df, no_da_source_df, no_da_target_df, fig_prefix):
    da_source_df_stand = StandardScaler().fit_transform(da_source_df)
    da_target_df_stand = StandardScaler().fit_transform(da_target_df)
    tsne = TSNE(n_components=2, n_jobs=-1, perplexity=5)
    da_source_components = tsne.fit_transform(da_source_df_stand)
    da_target_components = tsne.fit_transform(da_target_df_stand)

    no_da_source_df_stand = StandardScaler().fit_transform(no_da_source_df)
    no_da_target_df_stand = StandardScaler().fit_transform(no_da_target_df)
    tsne = TSNE(n_components=2, n_jobs=-1, perplexity=5)
    no_da_source_components = tsne.fit_transform(no_da_source_df_stand)
    no_da_target_components = tsne.fit_transform(no_da_target_df_stand)

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 16})
    plt.axis("off")

    lo = plt.scatter(
        da_source_components.T[0],
        da_source_components.T[1],
        color="black",
        alpha=1,
        marker="o",
        s=1,
    )
    ll = plt.scatter(
        da_target_components.T[0],
        da_target_components.T[1],
        color="red",
        alpha=0.5,
        marker="v",
        s=1,
    )
    # ly = plt.scatter(
    #     da_target_components[4268:].T[0],
    #     da_target_components[4268:].T[1],
    #     color="blue",
    #     alpha=0.5,
    #     marker="v",
    #     s=1,
    # )
    plt.legend(
        (lo, ll),
        ("Source domain", "Target domain"),
        scatterpoints=1,
        markerscale=6,
        loc="upper right",
        fontsize=14,
    )
    plt.title("Transfer Learning with Domain Adaptation")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.savefig(f"figures/{fig_prefix}_da.png")

    lo = plt.scatter(
        no_da_source_components.T[0],
        no_da_source_components.T[1],
        color="black",
        alpha=1,
        marker="o",
        s=1,
    )
    ll = plt.scatter(
        no_da_target_components.T[0],
        no_da_target_components.T[1],
        color="red",
        alpha=0.5,
        marker="v",
        s=1,
    )
    # ly = plt.scatter(
    #     no_da_target_components[4268:].T[0],
    #     no_da_target_components[4268:].T[1],
    #     color="blue",
    #     alpha=0.5,
    #     marker="v",
    #     s=1,
    # )
    plt.legend(
        (lo, ll),
        ("Source domain", "Target domain (Fake)", "Target domain (Non-fake)"),
        scatterpoints=1,
        markerscale=6,
        loc="upper right",
        fontsize=14,
    )
    plt.title("Transfer Learning without Domain Adaptation")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.savefig(f"figures/{fig_prefix}_no_da.png")
    plt.savefig(f"figures/{fig_prefix}_no_da.png")