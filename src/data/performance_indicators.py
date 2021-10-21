from bson.code import Code


def create_performance_indicators(matches_coll):
    map_function = Code("function () {"
                        "  var docId =  this._id;"
                        "  var totalRounds =  0;"
                        "  this.score.split(' / ').forEach(function(z) {"
                        "    totalRounds += parseInt(z);"
                        "  });"
                        "  this.teams.forEach(team => {"
                        "   Object.values(team).forEach(p => {"
                        "       if (p.hasOwnProperty('playerStats')===true && p.playerStats !== null) {"
                        "           var kills = parseFloat(p.playerStats.kills);"
                        "           var assists = parseFloat(p.playerStats.assists);"
                        "           var deaths = parseFloat(p.playerStats.deaths);"
                        "           var mvps = parseFloat(p.playerStats.mvps);"
                        "           var tripleKills = parseFloat(p.playerStats.tripleKills);"
                        "           var quadraKills = parseFloat(p.playerStats.quadraKills);"
                        "           var pentaKills = parseFloat(p.playerStats.pentaKills);"
                        "           emit('kills_pr', kills/totalRounds);"
                        "           emit('survived_pr', (totalRounds-deaths)/totalRounds);"
                        "           emit('assists_pr', assists/totalRounds);"
                        "           emit('multikills_rating_pr', (tripleKills*9+quadraKills*16+pentaKills*25)/totalRounds);"
                        "           emit('mvps_pr', mvps/totalRounds);"
                        "       } "
                        "   });"
                        "  });"
                        "}")

    reduce_function = Code("function (key, values) {"
                           "  var sum = 0;"
                           "  for (var i = 0; i < values.length; i++) {"
                           "    if(typeof values[i] == 'number'){"
                           "       sum += values[i];"
                           "    };"
                           "  }"
                           "  var mean = sum / values.length;"
                           "  var squaredDiffToMean = 0;"
                           "  for (var i = 0; i < values.length; i++) {"
                           "    if(typeof values[i] == 'number'){"
                           "       squaredDiffToMean += (values[i] - mean)*(values[i] - mean);"
                           "    };"
                           "  }"
                           "  var variance = squaredDiffToMean/values.length;"
                           "  var stdDev = Math.pow(variance, 1/2);"
                           "  return {mean, stdDev};"
                           "}")

    matches_coll.map_reduce(
        map_function,
        reduce_function,
        "performance_statistics")


# kills_pr = []
# survived_pr = []
# multikills_rating_pr = []
# assists = []
# assists_pr = []
# mvps_pr = []


# def get_average_stats(match):
#     team_rounds = [int(r) for r in match['score'].split("/")]
#     total_rounds = sum(team_rounds)

#     for team in (match["teamA"], match["teamB"]):
#         for player in team:
#             if(player['playerStats']==None):
#                 continue
#             stats = player['playerStats']

#             kills_pr.append(stats["kills"]/total_rounds)
#             survived_pr.append((total_rounds - stats['deaths'])/total_rounds)
#             multikills_rating_pr.append((stats['tripleKills']*9 +stats['quadraKills']*16 + stats['pentaKills']*25)/total_rounds)
#             assists_pr.append(stats['assists']/total_rounds)
#             mvps_pr.append(stats['mvps']/total_rounds)

# for data in read_data_iter("interim"):
#     data.apply(get_average_stats, axis=1)

# performance_statistics = {
#     'meanKPR': mean(kills_pr),
#     'stddevKPR': np.std(kills_pr),
#     'meanSPR': mean(survived_pr),
#     'stddevSPR': np.std(survived_pr),
#     'meanMKPR': mean(multikills_rating_pr),
#     'stddevMKPR': np.std(multikills_rating_pr),
#     'meanAPR': mean(assists_pr),
#     'stddevAPR': np.std(assists_pr),
#     'meanMVPPR': mean(mvps_pr),
#     'stddevMVPPR': np.std(mvps_pr)}

# # Store performance statistics
# with open(str(EXTERNAL_DATA_DIR) + '/performance_statistics.json', 'w') as fp:
#     json.dump(performance_statistics, fp)
