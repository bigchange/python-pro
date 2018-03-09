import MySQLdb
import json

def main():
    with open("./project_detail.txt", "w") as fp:
        total = 0
        # db = MySQLdb.connect("192.168.16.6", "idmg", "2EnQ8}QYgZ%8", "ai_pipeline")
        db = MySQLdb.connect("172.16.52.101", "pipeline", "Pipeline2018", "ai_pipeline")
        cursor = db.cursor()
        cursor.execute("SELECT project_detail from project_detail")
        print ("[SELECT]")
        while True:
            r = cursor.fetchone()
            if not r:
                break
            # arr = json.loads(r[0])
            fp.write("%s\n" % r[0])
            total += 1
        print("totally :%d" % (total))

if __name__ == '__main__':
    main()