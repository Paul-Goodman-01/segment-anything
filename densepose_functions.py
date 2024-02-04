
# Get rescaled image metrics
def getRescaledImageMetrics(image_metrics, new_w, new_h):
    new_image_metrics = image_metrics
    new_image_metrics['IM_SHAPE'] = [new_h, new_w]
    new_image_metrics['ASP_RATIO'] = new_w / new_h
    return new_image_metrics

# Get rescaled metrics for a single body part (if necessary)
# NB: Creates a copy of the original metrics and returns the altered copy
def getRescaledMetrics(metrics, new_w, new_h):
    new_metrics = metrics
    if new_metrics['NORMALISED'] == True and new_w > 0 and new_h > 0:
        new_metrics['AREAS'] = [x * (new_w * new_h) for x in new_metrics['AREAS']]
        new_metrics['BBOXS'] = [(round(y1 * new_h), round(x1 * new_w), round(y2 * new_h), round(x2 * new_w)) for y1, x1, y2, x2 in new_metrics['BBOXS']]
        new_metrics['CENTS'] = [(round(y * new_h), round(x * new_w)) for y, x in new_metrics['CENTS']]
        new_metrics['OOBB_CENTS'] = [[round(y * new_h), round(x * new_w)] for y, x in metrics['OOBB_CENTS']]       
        new_metrics['OOBBS'] = [[[round(y * new_h), round(x * new_w)] for y, x in regions] for regions in metrics['OOBBS']]           
        new_metrics['OOBB_MIDS'] = [[[round(y * new_h), round(x * new_w)] for y, x in regions] for regions in metrics['OOBB_MIDS']]           
        new_metrics['NORMALISED'] = False
    return new_metrics

#Get rescaled results at the image level
def getRescaledResults(results, new_w, new_h):
    new_results = {}
    if new_w > 0 and new_h > 0:
        new_results = results
        for item in results['PART_LIST']:
            new_part_results = getRescaledMetrics(results[item], new_w, new_h)
            new_results[item] = new_part_results
    #new_results['IM_SHAPE'] = [new_h, new_w]
    return new_results