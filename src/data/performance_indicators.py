from bson.code import Code

from src.db.connections import get_local_db

db = get_local_db()

matches_coll = db['match']
performance_stats_coll = db['performance_statistics']


def create_performance_indicators():
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


def load_average_indicators():
    kills = performance_stats_coll.find_one({"_id": "kills_pr"})
    survived = performance_stats_coll.find_one({"_id": "survived_pr"})
    multikills = performance_stats_coll.find_one(
        {"_id": "multikills_rating_pr"})
    assists = performance_stats_coll.find_one({"_id": "assists_pr"})
    mvps = performance_stats_coll.find_one({"_id": "mvps_pr"})
    stats = [kills, survived, multikills, assists, mvps]

    return [s["value"]["mean"] for s in stats]
